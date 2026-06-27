"""
The implementation of SAITS for the partially-observed time-series imputation task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _SAITS
from .data import DatasetForSAITS
from pypots.imputation.base import BaseNNImputer
from pypots.data.checking import key_in_data_set
from pypots.data.dataset import BaseDataset
from pypots.nn.modules.loss import Criterion, MAE, MSE
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer
from pypots.utils.logging import logger


# ============================================================================
# crazy-tries branch — Round 1: Harmonic-Subspace Refinement (HSR)
# ----------------------------------------------------------------------------
# Inference-time post-processing. For each (window, feature) it fits a smooth,
# periodic basis to the OBSERVED cells by ridge-regularised least squares, then
# blends that fit with the model output on the missing cells with a weight
# equal to the missing fraction. Classical regularised least-squares spectral
# analysis (harmonic regression). See CRAZY_TRIES_LOG.md, Round 1.
# ============================================================================
def _harmonic_basis(n_steps: int, n_harmonics: int, device, dtype) -> torch.Tensor:
    """Fixed design matrix Phi [T, P]: quadratic trend + Fourier harmonics."""
    t = torch.arange(n_steps, device=device, dtype=dtype)
    tn = t / max(n_steps - 1, 1) * 2.0 - 1.0          # normalised time in [-1, 1]
    cols = [torch.ones_like(tn), tn, tn * tn]         # constant + linear + quadratic trend
    for k in range(1, n_harmonics + 1):
        ang = 2.0 * math.pi * k * t / n_steps
        cols.append(torch.cos(ang))
        cols.append(torch.sin(ang))
    return torch.stack(cols, dim=1)                   # [T, P], P = 3 + 2*n_harmonics


def _hsr_refine(X, missing_mask, X_tilde_3, Phi, lam: float) -> torch.Tensor:
    """Ridge-fit Phi to observed cells per (batch, feature); blend with model.

    X, missing_mask, X_tilde_3 : [B, T, N]   (missing_mask: 1 = observed)
    Phi                        : [T, P]
    """
    m = missing_mask                                              # [B, T, N]
    mPhi = torch.einsum("btn,tp->btnp", m, Phi)                   # [B, T, N, P]
    A = torch.einsum("btnp,tq->bnpq", mPhi, Phi)                  # [B, N, P, P]
    rhs = torch.einsum("btnp,btn->bnp", mPhi, X)                  # [B, N, P]
    P = Phi.shape[1]
    eye = torch.eye(P, device=Phi.device, dtype=Phi.dtype)
    # the linear solve is tiny (P~27); run it on CPU for MPS/backend safety
    c = torch.linalg.solve((A + lam * eye).cpu(), rhs.cpu().unsqueeze(-1))
    c = c.squeeze(-1).to(X.device)                                # [B, N, P]
    recon = torch.einsum("tp,bnp->btn", Phi, c)                   # [B, T, N]
    w = 1.0 - m.mean(dim=1, keepdim=True)                         # [B, 1, N] missing fraction
    blended = (1.0 - w) * X_tilde_3 + w * recon
    return m * X + (1.0 - m) * blended                            # keep observed cells exact


# ============================================================================
# crazy-tries branch — Round 2: Whittaker-Henderson / HP-filter refinement
# ----------------------------------------------------------------------------
# Same blend-with-an-anchored-fit template as Round 1, but the fit is the
# non-parametric Whittaker-Henderson smoother (a.k.a. the Hodrick-Prescott
# filter): minimise data fidelity on observed cells + a second-difference
# (curvature) penalty. Positive-definite for any mask, so — unlike Round 1's
# Fourier basis — it cannot overfit/oscillate when observations are scarce.
# See CRAZY_TRIES_LOG.md, Round 2.
# ============================================================================
def _second_diff_matrix(n_steps: int, device, dtype) -> torch.Tensor:
    """(T-2) x T second-difference operator D."""
    T = n_steps
    D = torch.zeros(T - 2, T, device=device, dtype=dtype)
    idx = torch.arange(T - 2)
    D[idx, idx] = 1.0
    D[idx, idx + 1] = -2.0
    D[idx, idx + 2] = 1.0
    return D


def _whittaker_fit(X, missing_mask, alpha: float, eps: float = 1e-6,
                   chunk: int = 1024):
    """Whittaker-Henderson / HP smoother fit to the observed cells.

    X, missing_mask : [B, T, N]   (missing_mask: 1 = observed)
    Solves  z = (diag(m) + alpha*DtD + eps*I)^-1 diag(m) x  per (batch, feature).
    Returns the smoothed series z : [B, T, N].

    The B*N independent T x T systems are solved in chunks so memory stays
    bounded regardless of feature count (datasets here range 7 -> 862 feats).
    """
    B, T, N = X.shape
    D = _second_diff_matrix(T, X.device, X.dtype)                 # [T-2, T]
    # shared part of every system: alpha*DtD + eps*I  (on CPU for the solve)
    base = (alpha * (D.t() @ D)
            + eps * torch.eye(T, device=X.device, dtype=X.dtype)).cpu()
    m_flat = missing_mask.permute(0, 2, 1).reshape(B * N, T).cpu()     # [BN, T]
    rhs_flat = ((missing_mask * X).permute(0, 2, 1)
                .reshape(B * N, T, 1)).cpu()                          # [BN, T, 1]
    z_parts = []
    for s in range(0, B * N, chunk):
        mc = m_flat[s:s + chunk]                                      # [c, T]
        Ac = torch.diag_embed(mc) + base                              # [c, T, T]
        # CPU solve for backend safety (MPS lacks batched linalg.solve)
        zc = torch.linalg.solve(Ac, rhs_flat[s:s + chunk])            # [c, T, 1]
        z_parts.append(zc.squeeze(-1))
    z = torch.cat(z_parts, dim=0).reshape(B, N, T).to(X.device)
    return z.permute(0, 2, 1)                                         # [B, T, N]


def _whittaker_refine(X, missing_mask, X_tilde_3, alpha: float, eps: float = 1e-6):
    """HP-filter smoother fit to observed cells, blended with the model.

    Blend weight = per-window per-feature missing fraction (Round 2; blind).
    """
    m = missing_mask
    z = _whittaker_fit(X, m, alpha, eps)
    w = 1.0 - m.mean(dim=1, keepdim=True)                         # [B, 1, N] missing fraction
    blended = (1.0 - w) * X_tilde_3 + w * z
    return m * X + (1.0 - m) * blended                            # keep observed cells exact


class SAITS_MY(BaseNNImputer):
    """The PyTorch implementation of the SAITS model :cite:`du2023SAITS`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_layers :
        The number of layers in the 1st and 2nd DMSA blocks in the SAITS model.

    d_model :
        The dimension of the model's backbone.
        It is the input dimension of the multi-head DMSA layers.

    n_heads :
        The number of heads in the multi-head DMSA mechanism.
        ``d_model`` must be divisible by ``n_heads``, and the result should be equal to ``d_k``.

    d_k :
        The dimension of the `keys` (K) and the `queries` (Q) in the DMSA mechanism.
        ``d_k`` should be the result of ``d_model`` divided by ``n_heads``. Although ``d_k`` can be directly calculated
        with given ``d_model`` and ``n_heads``, we want it be explicitly given together with ``d_v`` by users to ensure
        users be aware of them and to avoid any potential mistakes.

    d_v :
        The dimension of the `values` (V) in the DMSA mechanism.

    d_ffn :
        The dimension of the layer in the Feed-Forward Networks (FFN).

    dropout :
        The dropout rate for all fully-connected layers in the model.

    attn_dropout :
        The dropout rate for DMSA.

    diagonal_attention_mask :
        Whether to apply a diagonal attention mask to the self-attention mechanism.
        If so, the attention layers will use DMSA. Otherwise, the attention layers will use the original.

    ORT_weight :
        The weight for the ORT loss.

    MIT_weight :
        The weight for the MIT loss.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    training_loss:
        The customized loss function designed by users for training the model.
        If not given, will use the default loss as claimed in the original paper.

    validation_metric:
        The customized metric function designed by users for validating the model.
        If not given, will use the default MSE metric.

    optimizer :
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers :
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device :
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy :
        The strategy to save model checkpoints. It has to be one of [None, "best", "better", "all"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.
        The "all" strategy will save every model after each epoch training.

    verbose :
        Whether to print out the training logs during the training process.

    mod_e :
        Course-project modification C2 — Variational Information Bottleneck
        (training-time). 1 enables it (default), 0 recovers the baseline.

    mod_l :
        Course-project modification I — Output Smoothing (inference-time,
        3-tap moving average on imputed cells). 1 enables it (default).

    mod_m :
        Course-project modification J — Median Filter (inference-time, 3-tap
        median on imputed cells). Alternative to ``mod_l``; if both are 1,
        ``mod_l`` takes precedence. Defaults to 0.
    """

    def __init__(
        self,
        loss_type: str,
        loss_weight: list,
        align_type: str,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float = 0,
        attn_dropout: float = 0,
        diagonal_attention_mask: bool = True,
        ORT_weight: int = 1,
        MIT_weight: int = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
        # ====================================================================
        # COURSE-PROJECT MODIFICATIONS — Mattivi & Feliu, TSA 2026
        # Three confirmed improvements to Glocal-IB, each a 0/1 toggle that
        # defaults to ON. Set all three to 0 to recover the GlocalIB_base
        # baseline. See RATIONALE.md / RESULTS.md.
        #   mod_e  C2: Variational Information Bottleneck (training-time;
        #             consumed inside _SAITS).
        #   mod_l  I:  Output Smoothing (inference-time; consumed in predict()).
        #   mod_m  J:  Median Filter (inference-time; consumed in predict()).
        #             Alternative to mod_l — if both are 1, mod_l wins.
        # ====================================================================
        mod_e: int = 1,
        mod_l: int = 1,
        mod_m: int = 0,
        # crazy-tries branch — inference-time experimental refinements.
        # Default OFF; mutually exclusive post-filters with mod_l/mod_m.
        # predict() precedence: mod_l > mod_m > mod_n > mod_o > mod_p.
        #   mod_n  R1: Harmonic-Subspace Refinement.
        #   mod_o  R2: Whittaker-Henderson / HP-filter refinement.
        #   mod_p  R3: Self-Supervised Held-Out Blend (HP fit + CV blend).
        mod_n: int = 0,
        mod_o: int = 0,
        mod_p: int = 0,
        hp_alpha: float = 10.0,
        use_real_xori_mask: bool = True,
        physical_constraints: "object | None" = None,
    ):
        super().__init__(
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        if d_model != n_heads * d_k:
            logger.warning(
                "‼️ d_model must = n_heads * d_k, it should be divisible by n_heads "
                f"and the result should be equal to d_k, but got d_model={d_model}, n_heads={n_heads}, d_k={d_k}"
            )
            d_model = n_heads * d_k
            logger.warning(
                f"⚠️ d_model is reset to {d_model} = n_heads ({n_heads}) * d_k ({d_k})"
            )

        self.n_steps = n_steps
        self.n_features = n_features
        # model hype-parameters
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.align_type = align_type
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        # Course-project modification flags (Mattivi & Feliu 2026).
        # mod_e is forwarded into _SAITS (training-time change); mod_l and
        # mod_m are inference-only and consumed in predict() below.
        self.mod_e = int(mod_e)
        self.mod_l = int(mod_l)
        self.mod_m = int(mod_m)
        self.mod_n = int(mod_n)         # crazy-tries R1: HSR (inference-time)
        self.mod_o = int(mod_o)         # crazy-tries R2: Whittaker/HP (inference-time)
        self.mod_p = int(mod_p)         # crazy-tries R3: SSHB (inference-time)
        self.hp_alpha = float(hp_alpha)

        # set up the model
        self.model = _SAITS(
            loss_type=self.loss_type,
            loss_weight=self.loss_weight,
            align_type=self.align_type,
            n_layers=self.n_layers,
            n_steps=self.n_steps,
            n_features=self.n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            diagonal_attention_mask=self.diagonal_attention_mask,
            ORT_weight=self.ORT_weight,
            MIT_weight=self.MIT_weight,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
            # Course-project modification C2 (VIB) — the only modification
            # that touches the training-time core; I/J act in predict().
            mod_e=self.mod_e,
            use_real_xori_mask=use_real_xori_mask,
            physical_constraints=physical_constraints,
        )
        self._print_model_size()
        self._send_model_to_given_device()

        # set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        indices, X, missing_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }
        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForSAITS(
            train_set, return_X_ori=False, return_y=False, file_type=file_type
        )
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForSAITS(
                val_set, return_X_ori=True, return_y=False, file_type=file_type
            )
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(
            confirm_saving=self.model_saving_strategy == "best"
        )

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        diagonal_attention_mask: bool = True,
        return_latent_vars: bool = False,
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if test_set is a path string.

        diagonal_attention_mask :
            Whether to apply a diagonal attention mask to the self-attention mechanism in the testing stage.

        return_latent_vars :
            Whether to return the latent variables in SAITS, e.g. attention weights of two DMSA blocks and
            the weight matrix from the combination block, etc.

        Returns
        -------
        file_type :
            The dictionary containing the clustering results and latent variables if necessary.

        """
        self.model.eval()  # set the model to evaluation mode
        # Step 1: wrap the input data with classes Dataset and DataLoader
        test_set = BaseDataset(
            test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []
        enc_out_collector = []
        first_DMSA_attn_weights_collector = []
        second_DMSA_attn_weights_collector = []
        combining_weights_collector = []

        # Step 2: process the data with the model
        for idx, data in enumerate(test_loader):
            inputs = self._assemble_input_for_testing(data)
            results = self.model.forward(inputs, diagonal_attention_mask)

            # ================================================================
            # COURSE-PROJECT MODIFICATIONS I & J — inference-time post-filters
            # ----------------------------------------------------------------
            # BASELINE: the imputation is `results["imputed_data"]` directly
            #   (observed cells kept, missing cells filled with the model
            #   output X_tilde_3) — see the final `else` branch.
            # IMPROVEMENT I (mod_l): smooth X_tilde_3 with a 3-tap moving
            #   average before re-inserting the observed cells — a local
            #   time-series smoothness prior.
            # IMPROVEMENT J (mod_m): use a 3-tap median filter instead. This
            #   is an ALTERNATIVE to I; when both flags are on, I takes
            #   precedence (the `elif` below).
            # Both act ONLY on the imputed positions; observed cells are
            # always kept at their true value X. See RATIONALE.md.
            # ================================================================
            if self.mod_l:
                # --- Improvement I: Output Smoothing (3-tap moving average) ---
                mm = inputs["missing_mask"]
                X_tilde_3 = results["X_tilde_3"]   # [B, T, N]
                B, T, N = X_tilde_3.shape
                # reshape to [B*N, 1, T] for a 1-D conv along the time axis
                x_flat = X_tilde_3.permute(0, 2, 1).reshape(B * N, 1, T)
                kernel = torch.ones(1, 1, 3, device=X_tilde_3.device) / 3.0
                x_sm = torch.nn.functional.conv1d(x_flat, kernel, padding=1)
                X_tilde_sm = x_sm.reshape(B, N, T).permute(0, 2, 1)
                refined = mm * inputs["X"] + (1.0 - mm) * X_tilde_sm
                imputation_collector.append(refined)
            elif self.mod_m:
                # --- Improvement J: Median Filter (3-tap median) ---
                # The median is the optimal location estimator under a Laplace
                # likelihood — i.e. it matches the MAE training loss — and is
                # robust to the occasional outlier prediction.
                mm = inputs["missing_mask"]
                X_tilde_3 = results["X_tilde_3"]   # [B, T, N]
                # replicate the boundary, then median over (t-1, t, t+1)
                X_pad = torch.nn.functional.pad(X_tilde_3, (0, 0, 1, 1), mode="replicate")
                stacked = torch.stack(
                    [X_pad[:, 0:-2, :], X_pad[:, 1:-1, :], X_pad[:, 2:, :]],
                    dim=0,
                )
                X_tilde_med = stacked.median(dim=0).values
                refined = mm * inputs["X"] + (1.0 - mm) * X_tilde_med
                imputation_collector.append(refined)
            elif self.mod_n:
                # --- crazy-tries R1: Harmonic-Subspace Refinement (HSR) ---
                # Ridge-fit a trend+Fourier basis to the observed (ground-truth)
                # cells of each window/feature, then blend that fit with the
                # model output on the missing cells with weight = missing
                # fraction. See _hsr_refine / CRAZY_TRIES_LOG.md Round 1.
                mm = inputs["missing_mask"]
                X_tilde_3 = results["X_tilde_3"]            # [B, T, N]
                T = X_tilde_3.shape[1]
                Phi = _harmonic_basis(T, 12, X_tilde_3.device, X_tilde_3.dtype)
                refined = _hsr_refine(inputs["X"], mm, X_tilde_3, Phi, lam=1.0)
                imputation_collector.append(refined)
            elif self.mod_o:
                # --- crazy-tries R2: Whittaker-Henderson / HP refinement ---
                mm = inputs["missing_mask"]
                X_tilde_3 = results["X_tilde_3"]            # [B, T, N]
                refined = _whittaker_refine(
                    inputs["X"], mm, X_tilde_3, alpha=self.hp_alpha
                )
                imputation_collector.append(refined)
            elif self.mod_p:
                # --- crazy-tries R3: Self-Supervised Held-Out Blend (SSHB) ---
                # Blend the HP smoother with the model on the missing cells,
                # but pick the per-window blend weight by honest held-out
                # cross-validation: hide 25% of the observed cells, see which
                # of {model, smoother} predicts them better, weight
                # accordingly. See CRAZY_TRIES_LOG.md, Round 3.
                mm = inputs["missing_mask"]                 # [B, T, N], 1=observed
                X = inputs["X"]
                B, T, N = X.shape
                X_tilde_3_A = results["X_tilde_3"]          # model, full observed data

                # (1) hold out a deterministic 10% of the observed cells.
                # 10% (not 25%) so the second forward pass sees almost the
                # same missing rate as the real one — the held-out model
                # prediction stays honest even at mr=0.9.
                gen = torch.Generator(device="cpu").manual_seed(20260515)
                rand = torch.rand(B, T, N, generator=gen).to(mm.device)
                H = ((mm > 0.5) & (rand < 0.10)).to(X.dtype)   # held-out observed cells
                keep = 1.0 - H                                 # 1 = still visible
                mm_ho = mm * keep
                X_ho = X * keep

                # (2) honest model imputation of H — second forward pass
                results_B = self.model.forward(
                    {"X": X_ho, "missing_mask": mm_ho}, diagonal_attention_mask
                )
                X_tilde_3_B = results_B["X_tilde_3"]

                # (3) honest smoother imputation of H, and full-data fit
                z_ho = _whittaker_fit(X_ho, mm_ho, self.hp_alpha)
                z_full = _whittaker_fit(X, mm, self.hp_alpha)

                # (4) per-FEATURE blend weight by least squares on H (pooled
                #     over the batch and time so the estimate is robust even
                #     when each window holds only ~10 observed cells):
                #     w* = argmin_w sum_H ((1-w)*model + w*smoother - truth)^2
                a = (z_ho - X_tilde_3_B) * H                  # smoother - model, on H
                r = (X - X_tilde_3_B) * H                     # truth - model, on H
                num = (a * r).sum(dim=(0, 1))                 # [N]
                den = (a * a).sum(dim=(0, 1)) + 1e-6          # [N]
                w = (num / den).clamp(0.0, 1.0).view(1, 1, N) # model (w=0) is the floor

                # (5) apply w* to the truly-missing cells
                blended = (1.0 - w) * X_tilde_3_A + w * z_full
                refined = mm * X + (1.0 - mm) * blended
                imputation_collector.append(refined)
            else:
                # --- baseline path: no inference-time post-filtering ---
                imputation_collector.append(results["imputed_data"])
            enc_out_collector.append(results["X_obs_p_contras"])

            if return_latent_vars:
                first_DMSA_attn_weights_collector.append(
                    results["first_DMSA_attn_weights"].cpu().numpy()
                )
                second_DMSA_attn_weights_collector.append(
                    results["second_DMSA_attn_weights"].cpu().numpy()
                )
                combining_weights_collector.append(
                    results["combining_weights"].cpu().numpy()
                )

        # Step 3: output collection and return
        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }

        if return_latent_vars:
            latent_var_collector = {
                "first_DMSA_attn_weights": np.concatenate(
                    first_DMSA_attn_weights_collector
                ),
                "second_DMSA_attn_weights": np.concatenate(
                    second_DMSA_attn_weights_collector
                ),
                "combining_weights": np.concatenate(combining_weights_collector),
            }
            result_dict["latent_vars"] = latent_var_collector

        result_dict["enc_out"] = torch.cat(enc_out_collector).cpu().detach().numpy()

        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["imputation"]

    def get_all_info(
        self,
        data_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        result_dict = self.predict(data_set, file_type=file_type)
        return result_dict
