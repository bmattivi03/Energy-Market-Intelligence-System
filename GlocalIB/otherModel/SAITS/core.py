"""
The core wrapper assembles the submodules of SAITS imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn

from pypots.nn.modules import ModelCore
from pypots.nn.modules.loss import Criterion

from .backbone import BackboneSAITS

import subprocess
import os
from transformers import AutoModelForCausalLM

from ..loss import (
    MyContrastiveLoss_v1,
    MyContrastiveLoss_v2,
    MyAlignmentLoss,
)

import muyi.utils as muu


class _SAITS(ModelCore):
    def __init__(
        self,
        loss_type: str,
        loss_weight: list,
        align_type: str,
        n_layers: int,
        n_steps: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        diagonal_attention_mask: bool,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
        # ====================================================================
        # COURSE-PROJECT MODIFICATION - Mattivi & Feliu, TSA 2026
        # mod_e toggles improvement C2 (Variational Information Bottleneck), a
        # training-time change consumed inside this _SAITS core. Default ON;
        # set to 0 to recover the GlocalIB_base baseline. The other two
        # confirmed improvements (Output Smoothing, Median Filter) are
        # inference-only and live in SAITS_MY.predict, not here.
        # See RATIONALE.md.
        # ====================================================================
        mod_e: int = 1,
        use_real_xori_mask: bool = True,
        physical_constraints: "object | None" = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.align_type = align_type
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.encoder = BackboneSAITS(
            n_steps,
            n_features,
            n_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )

        self.contrastive_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            # nn.BatchNorm1d(n_steps),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        if self.align_type == "FM_align":
            muu.color_print(
                f"!!!!!!!!!! Using foundation model (Time-MoE 50M Frozen) !!!!!!!!!!"
            )
            self.foundation_model = AutoModelForCausalLM.from_pretrained(
                "../Time-MoE/TimeMoE-50M",
                device_map="cuda",
                trust_remote_code=True,
            )
            self.foundation_model.requires_grad_(False)
            self.alignment_projection = nn.Sequential(
                nn.Linear(d_model, n_features),
                nn.ReLU(),
                nn.Linear(n_features, n_features),
            )
        else:
            self.foundation_model = None
            self.alignment_projection = None

        self.contrastive_loss_v1 = MyContrastiveLoss_v1()
        self.contrastive_loss_v2 = MyContrastiveLoss_v2()
        self.alignment_loss = MyAlignmentLoss()

        # ====================================================================
        # COURSE-PROJECT MODIFICATION C2 - Variational Information Bottleneck
        # (Mattivi & Feliu 2026; Alemi et al. 2017 "Deep Variational
        # Information Bottleneck").
        #
        # When self.mod_e == 1, the deterministic `contrastive_projection`
        # built above is replaced by a STOCHASTIC projection: the two heads
        # below emit the mean and log-variance of a Gaussian posterior over
        # the contrastive embedding. forward() samples it with the
        # reparameterization trick and calc_criterion() adds a KL-to-prior
        # term - turning the contrastive head into an information bottleneck
        # that compresses the representation.
        #
        # The heads are constructed UNCONDITIONALLY (even when mod_e == 0) so
        # the module-init RNG order - and therefore the initial values of
        # every parameter - is identical whether the flag is on or off. This
        # is what makes the baseline path (mod_e=0) bit-for-bit reproducible.
        # ====================================================================
        self.mod_e = int(mod_e)
        self.use_real_xori_mask = use_real_xori_mask
        # Optional auxiliary physical-bound penalty on imputed values.
        # If None, no penalty is added (preserves original training dynamics).
        self.physical_constraints = physical_constraints
        self.proj_mu = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.proj_logvar = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        inputs: dict,
        diagonal_attention_mask: bool = True,
        # [MOD-v2-2: Accept calc_criterion kwarg from pypots 0.13]
        # Compatibility patch - pypots calls model(inputs, calc_criterion=True)
        # during training; the inner _SAITS originally did not accept this.
        calc_criterion: bool = False,
    ) -> dict:
        if calc_criterion:
            return self.calc_criterion(inputs)
        if self.training:
            X, X_ori, missing_mask = (
                inputs["X"],
                inputs["X_ori"],
                inputs["missing_mask"],
            )  # [B, T, N]
            indicating_mask = inputs.get("indicating_mask")  # [B, T, N], optional
        else:
            X, missing_mask = (
                inputs["X"],
                inputs["missing_mask"],
            )  # [B, T, N]
            indicating_mask = None

        if self.align_type == "FM_align":
            X_foundation = self.foundation_model.generate(
                X.reshape(-1, self.n_features), max_new_tokens=self.n_features
            )[:, -self.n_features :].reshape(X.shape[0], -1, self.n_features)
        else:
            X_foundation = None

        # determine the attention mask
        if (self.training and self.diagonal_attention_mask) or (
            (not self.training) and diagonal_attention_mask
        ):
            diagonal_attention_mask = (1 - torch.eye(self.n_steps)).to(X.device)
            # then broadcast on the batch axis
            diagonal_attention_mask = diagonal_attention_mask.unsqueeze(0)
        else:
            diagonal_attention_mask = None

        # SAITS processing on the (corrupted) input X
        (
            enc_output_1,
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        ) = self.encoder(X, missing_mask, diagonal_attention_mask)

        X_obs_z_contras = enc_output_1
        # ====================================================================
        # COURSE-PROJECT MODIFICATION C2 - Variational Information Bottleneck
        # --------------------------------------------------------------------
        # BASELINE (mod_e == 0): X_obs_p_contras is the output of a plain
        #   deterministic MLP - see the `else` branch below.
        # IMPROVEMENT (mod_e == 1): the masked-view embedding is mapped to a
        #   Gaussian posterior N(mu, exp(logvar)) by two MLP heads. During
        #   training we draw a sample with the reparameterization trick
        #   (mu + sigma * eps) so gradients flow; at eval time we use the
        #   mean (deterministic). logvar is clamped for numerical stability.
        #   A KL-to-prior penalty is added in calc_criterion() - together
        #   this makes the contrastive projection an information bottleneck
        #   that compresses the representation and reduces over-fitting.
        # ====================================================================
        VIB_mu = None
        VIB_logvar = None
        if self.mod_e:
            VIB_mu = self.proj_mu(X_obs_z_contras)
            VIB_logvar = self.proj_logvar(X_obs_z_contras).clamp(min=-8.0, max=4.0)
            if self.training:
                eps = torch.randn_like(VIB_mu)
                X_obs_p_contras = VIB_mu + torch.exp(0.5 * VIB_logvar) * eps
            else:
                X_obs_p_contras = VIB_mu
        else:
            # --- baseline path: deterministic contrastive projection ---
            X_obs_p_contras = self.contrastive_projection(X_obs_z_contras)

        # Alignment branch ----------------------------------------------------
        X_ori_z_contras = None
        X_ori_p_contras = None
        if self.training:
            # PROJECT MERGE: use_real_xori_mask picks the genuine X_ori
            # observation mask (missing_mask + indicating_mask) instead of
            # all-ones, so the contrastive target is not partly noise.
            # X_ori has its NaNs replaced with 0 in the data loader; an
            # all-ones mask would claim those zero-filled cells are observed.
            if self.use_real_xori_mask and indicating_mask is not None:
                xori_mask = (missing_mask + indicating_mask).clamp(max=1.0)
            else:
                xori_mask = torch.ones_like(missing_mask)
            X_ori_z_contras = self.encoder(
                X_ori, xori_mask, diagonal_attention_mask
            )[0]
            # COURSE-PROJECT MODIFICATION C2 (VIB): project the complete-view
            # embedding with the stochastic head's MEAN (proj_mu) when mod_e is
            # on, so the contrastive target lives in the same projected space
            # as the sampled masked-view embedding. mod_e == 0 keeps the
            # baseline's deterministic contrastive_projection.
            if self.mod_e:
                X_ori_p_contras = self.proj_mu(X_ori_z_contras)
            else:
                # --- baseline path ---
                X_ori_p_contras = self.contrastive_projection(X_ori_z_contras)

        if self.align_type == "FM_align":
            X_ori_align = self.alignment_projection(X_obs_z_contras)  # [B, N, T]
        else:
            X_ori_align = None

        # replace the observed part with values from X
        imputed_data = missing_mask * X + (1 - missing_mask) * X_tilde_3

        results = {
            "first_DMSA_attn_weights": first_DMSA_attn_weights,
            "second_DMSA_attn_weights": second_DMSA_attn_weights,
            "combining_weights": combining_weights,
            "imputed_data": imputed_data,
            "X_tilde_1": X_tilde_1,
            "X_tilde_2": X_tilde_2,
            "X_tilde_3": X_tilde_3,
            "enc_out": enc_output_1,
            "X_obs_z_contras": X_obs_z_contras,
            "X_obs_p_contras": X_obs_p_contras,
            "X_ori_z_contras": X_ori_z_contras,
            "X_ori_p_contras": X_ori_p_contras,
            "X_foundation": X_foundation,
            "X_ori_align": X_ori_align,
            # COURSE-PROJECT MODIFICATION C2 (VIB): Gaussian posterior params,
            # both None when mod_e == 0.
            "VIB_mu": VIB_mu,
            "VIB_logvar": VIB_logvar,
        }

        return results

    def calc_criterion(self, inputs: dict) -> dict:
        results = self.forward(inputs)
        X_tilde_1, X_tilde_2, X_tilde_3 = (
            results["X_tilde_1"],
            results["X_tilde_2"],
            results["X_tilde_3"],
        )
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if (
            self.training
        ):  # if in the training mode (the training stage), return loss result from training_loss
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]

            # ORT loss (observed reconstruction) - unchanged.
            ORT_loss = 0
            ORT_loss += self.training_loss(X_tilde_1, X, missing_mask)
            ORT_loss += self.training_loss(X_tilde_2, X, missing_mask)
            ORT_loss += self.training_loss(X_tilde_3, X, missing_mask)
            ORT_loss /= 3
            ORT_loss = self.ORT_weight * ORT_loss

            # MIT loss (Masked Imputation Task) - baseline Glocal-IB term.
            MIT_loss = self.MIT_weight * self.training_loss(
                X_tilde_3, X_ori, indicating_mask
            )

            loss = ORT_loss + MIT_loss
            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss

            # Alignment branch - baseline Glocal-IB contrastive losses.
            # With VIB (mod_e) on, these losses operate on the stochastic
            # projection's sample/mean transparently - no branching needed.
            results["Contrastive_loss_v1"] = self.contrastive_loss_v1(
                results=results
            )
            results["Contrastive_loss_v2"] = self.contrastive_loss_v2(
                results=results
            )

            if self.align_type == "FM_align":
                results["Alignment_loss"] = self.alignment_loss(results=results)
            else:
                results["Alignment_loss"] = 0

            # COURSE-PROJECT MODIFICATION C2 (VIB): KL-to-prior on the
            # stochastic projection. KL[N(mu, sigma^2) || N(0, I)] pulls the
            # posterior toward a unit Gaussian, which is the "compression"
            # half of the information bottleneck. It is carried by the
            # kl_weight slot (loss_weight[1]) - dormant at 1e-6 in the
            # baseline, set to 1e-3 here (see utils.py --kl_weight).
            if self.mod_e and results.get("VIB_mu") is not None:
                mu, logvar = results["VIB_mu"], results["VIB_logvar"]
                kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
                results["VIB_kl"] = kl
            else:
                results["VIB_kl"] = torch.tensor(0.0, device=X.device)

            results["loss"] = 0.0
            if "1" in self.loss_type:
                results["loss"] += loss * self.loss_weight[0]
            # COURSE-PROJECT MODIFICATION C2 (VIB): add the KL term to the
            # total loss only when mod_e is on (baseline keeps this slot
            # effectively dormant).
            if "2" in self.loss_type and self.mod_e:
                results["loss"] += results["VIB_kl"] * self.loss_weight[1]
            if "3" in self.loss_type:
                if self.align_type == "contras_1":
                    results["loss"] += (
                        results["Contrastive_loss_v1"] * self.loss_weight[2]
                    )
                elif self.align_type == "contras_2":
                    results["loss"] += (
                        results["Contrastive_loss_v2"] * self.loss_weight[2]
                    )
                elif self.align_type == "FM_align":
                    results["loss"] += results["Alignment_loss"] * self.loss_weight[2]

            # PROJECT MERGE: optional auxiliary physical-bound penalty on the
            # imputed values. Inactive when physical_constraints is None.
            if self.physical_constraints is not None:
                from .physical_loss import physical_constraint_loss
                phys = physical_constraint_loss(
                    results["imputed_data"], self.physical_constraints
                )
                results["Physical_loss"] = phys
                results["loss"] = results["loss"] + phys
        else:  # if in the eval mode (the validation stage), return metric result from validation_metric
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            results["metric"] = self.validation_metric(
                X_tilde_3, X_ori, indicating_mask
            )

        return results
