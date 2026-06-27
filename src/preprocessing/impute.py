import sys
import pathlib

# Vendored Glocal-IB code lives here; it provides ``otherModel.SAITS`` (the
# imputer model). Append (not insert-at-0) so it does NOT shadow the project's
# own same-named top-level packages — GlocalIB ships ``utils`` and ``data``
# packages that would otherwise mask ``src/utils`` and ``src/data``.
# ``otherModel`` exists only in GlocalIB, so appending still resolves it.
_GLOCAL_ROOT = pathlib.Path(__file__).parents[2] / "GlocalIB"
if str(_GLOCAL_ROOT) not in sys.path:
    sys.path.append(str(_GLOCAL_ROOT))

import numpy as np
import pandas as pd

# Canonical reproducibility helpers. Re-exported at module scope so existing
# call sites (and any test importing ``preprocessing.impute.set_seed``) keep
# working unchanged.
from utils.reproducibility import select_device, set_seed  # noqa: F401

NUCLEAR_SHUTDOWN_DATE = pd.Timestamp("2023-04-15", tz="UTC")
TRAIN_END = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
VAL_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
T = 168
STRIDE_TRAIN = 24
STRIDE_INFER = 24
DEFAULT_SEED = 42
STRUCTURAL_ZERO_COL = "gen_fossil_coal_gas_structural_zero"


class NaNScaler:
    """StandardScaler that ignores NaN during fit and preserves NaN during transform."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, arr: np.ndarray) -> "NaNScaler":
        assert arr.ndim == 2, f"NaNScaler.fit expects 2D array, got shape {arr.shape}"
        self.mean_ = np.nanmean(arr, axis=0)
        self.scale_ = np.nanstd(arr, axis=0, ddof=0)
        self.scale_[(self.scale_ == 0) | np.isnan(self.scale_)] = 1.0
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean_) / self.scale_

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.scale_ + self.mean_


def apply_domain_fixes(
    df: pd.DataFrame,
    *,
    add_structural_zero_indicator: bool = False,
) -> pd.DataFrame:
    """Apply domain-specific fixes to generation and fuel columns.

    Rules:
    - gen_fossil_coal_gas: zero-fill all NaN (structurally absent from ENTSO-E)
    - gen_nuclear: zero-fill NaN only where index > NUCLEAR_SHUTDOWN_DATE (2023-04-15)
    - carbon_ets: untouched (will be filled separately)

    If ``add_structural_zero_indicator`` is True, also appends a
    ``gen_fossil_coal_gas_structural_zero`` float32 indicator (1.0 where the
    raw value was NaN). This is what downstream models should mask from loss.
    Does not modify the input DataFrame.
    """
    out = df.copy()
    coal_gas_was_nan = out["gen_fossil_coal_gas"].isna()
    out["gen_fossil_coal_gas"] = out["gen_fossil_coal_gas"].fillna(0.0)
    post_shutdown = out.index > NUCLEAR_SHUTDOWN_DATE
    out.loc[post_shutdown, "gen_nuclear"] = (
        out.loc[post_shutdown, "gen_nuclear"].fillna(0.0)
    )
    if add_structural_zero_indicator:
        out[STRUCTURAL_ZERO_COL] = coal_gas_was_nan.astype(np.float32).to_numpy()
    return out


def make_windows(arr: np.ndarray, T: int, stride: int) -> tuple:
    """Slice [total, N] into overlapping [B, T, N] windows of dtype float32."""
    total, N = arr.shape
    starts = np.arange(0, total - T + 1, stride)
    windows = np.stack([arr[s : s + T] for s in starts]).astype(np.float32)
    return windows, starts


def aggregate_predictions(
    predictions: np.ndarray,
    starts: np.ndarray,
    total_len: int,
    N: int,
    T: int,
) -> np.ndarray:
    """Average overlapping window predictions into a [total_len, N] array."""
    result = np.zeros((total_len, N), dtype=np.float64)
    counts = np.zeros((total_len, 1), dtype=np.float64)
    for i, s in enumerate(starts):
        result[s : s + T] += predictions[i]
        counts[s : s + T] += 1
    return result / np.maximum(counts, 1)


def recompute_net_imports(df: pd.DataFrame) -> pd.DataFrame:
    """Compute net import columns from cross-border flow columns.

    Computes:
    - net_import_FR = FR_to_DELU - DELU_to_FR
    - net_import_AT = AT_to_DELU - DELU_to_AT
    - net_import_CH = CH_to_DELU - DELU_to_CH

    Returns a new DataFrame with the three net import columns appended.
    Does not modify the input DataFrame.
    """
    df = df.copy()
    df["net_import_FR"] = df["FR_to_DELU"] - df["DELU_to_FR"]
    df["net_import_AT"] = df["AT_to_DELU"] - df["DELU_to_AT"]
    df["net_import_CH"] = df["CH_to_DELU"] - df["DELU_to_CH"]
    return df


def build_train_val_sets(
    scaled_arr: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    val_mask_ratio: float = 0.20,
    seed: int = DEFAULT_SEED,
) -> tuple:
    """Slice scaled array into SAITS_MY-compatible train and val dicts.

    The val set gets MCAR masking at ``val_mask_ratio`` so the imputation MIT
    validation loss is meaningful. ``seed`` controls the MCAR mask placement
    so successive runs produce identical splits.
    """
    from pygrinder import mcar

    train_mask = index <= TRAIN_END
    val_mask = (index > TRAIN_END) & (index <= VAL_END)

    train_arr = scaled_arr[train_mask]
    val_arr = scaled_arr[val_mask]

    train_windows, _ = make_windows(train_arr, T=T, stride=STRIDE_TRAIN)
    val_windows, _ = make_windows(val_arr, T=T, stride=STRIDE_TRAIN)

    # Seed numpy RNG so pygrinder.mcar produces deterministic masks.
    np.random.seed(seed)
    val_masked = mcar(val_windows.copy(), p=val_mask_ratio)

    train_set = {"X": train_windows}
    val_set = {"X": val_masked, "X_ori": val_windows}
    return train_set, val_set


def train_model(
    train_set: dict,
    val_set: dict,
    n_features: int,
    saving_path: str,
    *,
    n_layers: int = 4,
    d_model: int = 512,
    n_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    d_ffn: int = 1024,
    dropout: float = 0.1,
    batch_size: int = 32,
    epochs: int = 300,
    patience: int = 50,
    device: str | None = None,
    use_real_xori_mask: bool = True,
    physical_constraints: object | None = None,
    mod_p: int = 0,
) -> object:
    """Instantiate and train SAITS_MY (Glocal-IB enhanced SAITS).

    All architecture and training hyperparameters are exposed as keyword args
    so the same pipeline can be re-run with smaller models, shorter training,
    or with the new ``physical_constraints`` auxiliary loss enabled.
    """
    from otherModel.SAITS.model import SAITS_MY

    pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)
    device = select_device(prefer=device)

    model = SAITS_MY(
        loss_type="13",
        loss_weight=[1.0, 0.0, 0.1],
        align_type="contras_1",
        n_steps=T,
        n_features=n_features,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ffn=d_ffn,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        device=device,
        saving_path=saving_path,
        model_saving_strategy="best",
        use_real_xori_mask=use_real_xori_mask,
        physical_constraints=physical_constraints,
        mod_e=1,
        mod_l=0,  # disable output-smoothing branch so it does not shadow mod_p
        mod_m=0,  # disable median-filter branch (predict precedence: l>m>n>o>p)
        mod_p=mod_p,
    )
    model.fit(train_set, val_set)
    return model


def impute_full_dataset(
    model: object,
    scaled_arr: np.ndarray,
) -> np.ndarray:
    """Run sliding window inference (stride=24) and average overlapping predictions."""
    total, N = scaled_arr.shape
    windows, starts = make_windows(scaled_arr, T=T, stride=STRIDE_INFER)
    test_set = {"X": windows}

    predictions = model.impute(test_set)  # [B, T, N]

    imputed = aggregate_predictions(predictions, starts, total_len=total, N=N, T=T)

    # write back observed values — never overwrite known data
    observed = ~np.isnan(scaled_arr)
    imputed[observed] = scaled_arr[observed]

    return imputed


def main(
    *,
    raw_path: pathlib.Path | None = None,
    out_path: pathlib.Path | None = None,
    ckpt_path: str | None = None,
    apply_constraints: bool = False,
    use_physical_loss: bool = False,
    skip_train: bool = False,
    epochs: int = 300,
    patience: int = 50,
    n_layers: int = 4,
    d_model: int = 512,
    seed: int = DEFAULT_SEED,
    sshb: bool = False,
):
    """End-to-end imputation pipeline.

    Parameters
    ----------
    apply_constraints : if True, applies post-hoc physical-bound clipping
        (`physical_constraints.apply_all`) before saving. Default False to
        preserve current behavior.
    use_physical_loss : if True, builds a PhysicalConstraintSpec from the
        scaler and passes it to SAITS_MY as an auxiliary training loss.
    skip_train : if True, loads an existing checkpoint and only re-runs
        inference + post-processing (useful when only the constraints change).
    sshb : if True, enables SSHB (mod_p) inference-time refinement on the
        improved Glocal-IB imputer. VIB (mod_e) is always on.
    """
    import torch

    mod_p = 1 if sshb else 0

    set_seed(seed)
    project_root = pathlib.Path(__file__).parents[2]
    raw_path = raw_path or project_root / "data" / "processed" / "emis_raw.parquet"
    out_path = out_path or project_root / "data" / "processed" / "emis_imputed.parquet"
    ckpt_path = ckpt_path or str(project_root / "checkpoints" / "imputer_saits")
    device = select_device()
    print(f"[impute] seed={seed} device={device}")

    df = pd.read_parquet(raw_path)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    nan_before = df.isnull().sum()
    print(f"\nNaN before fixes:\n{nan_before[nan_before > 0]}\n")

    df_fixed = apply_domain_fixes(df)

    train_mask = df_fixed.index <= TRAIN_END
    arr = df_fixed.values.astype(np.float64)
    scaler = NaNScaler().fit(arr[train_mask])
    scaled = scaler.transform(arr)

    physical_spec = None
    if use_physical_loss:
        from otherModel.SAITS.physical_loss import build_spec_from_scaler
        from preprocessing.imputation_eval import (
            CROSS_BORDER_PAIRS, GENERATION_NONNEG_COLS,
        )
        physical_spec = build_spec_from_scaler(
            column_names=list(df_fixed.columns),
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            nonneg_columns=GENERATION_NONNEG_COLS,
            flow_pairs=CROSS_BORDER_PAIRS,
            weight=0.05,
        )
        print(f"Physical-loss spec: {len(physical_spec.nonneg_indices)} non-neg cols, "
              f"{len(physical_spec.flow_pairs)} flow pairs")

    if not skip_train:
        train_set, val_set = build_train_val_sets(scaled, df_fixed.index, seed=seed)
        print(f"Train windows: {train_set['X'].shape} | Val windows: {val_set['X'].shape}\n")
        print(f"Training SAITS_MY (Glocal-IB) on {device}…")
        model = train_model(
            train_set, val_set, n_features=df_fixed.shape[1], saving_path=ckpt_path,
            epochs=epochs, patience=patience, n_layers=n_layers, d_model=d_model,
            device=device,
            physical_constraints=physical_spec,
            mod_p=mod_p,
        )
        if device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        from otherModel.SAITS.model import SAITS_MY
        # Construct model with same arch and load best checkpoint
        model = SAITS_MY(
            loss_type="13", loss_weight=[1.0, 0.0, 0.1], align_type="contras_1",
            n_steps=T, n_features=df_fixed.shape[1],
            n_layers=n_layers, d_model=d_model, n_heads=8, d_k=64, d_v=64, d_ffn=1024,
            dropout=0.1, batch_size=32, epochs=1, patience=1,
            device=device, saving_path=ckpt_path,
            mod_e=1, mod_l=0, mod_m=0, mod_p=mod_p,
        )
        # Find the best checkpoint produced by a previous fit. Match the model
        # file by name (SAITS_MY.pypots) — a bare "*.pypots" glob also catches
        # pypots' tensorboard event files (events.*.pypots), and sorted()[-1]
        # would then load the event file instead of the model.
        ckpts = sorted(pathlib.Path(ckpt_path).glob("**/SAITS_MY.pypots"))
        if not ckpts:
            raise FileNotFoundError(f"No SAITS checkpoint found under {ckpt_path}")
        model.load(str(ckpts[-1]))

    print("\nRunning full-dataset sliding-window inference…")
    imputed_scaled = impute_full_dataset(model, scaled)

    imputed = scaler.inverse_transform(imputed_scaled)
    df_imputed = pd.DataFrame(imputed, index=df_fixed.index, columns=df_fixed.columns)

    carbon_nan = df["carbon_ets"].isna()
    df_imputed.loc[carbon_nan, "carbon_ets"] = np.nan

    df_imputed = recompute_net_imports(df_imputed)

    if apply_constraints:
        from preprocessing.physical_constraints import apply_all
        before_cols = set(df_imputed.columns)
        df_imputed = apply_all(df_imputed, raw_df=df)
        new_cols = set(df_imputed.columns) - before_cols
        print(f"Applied physical constraints (added: {sorted(new_cols)})")

    nan_after = df_imputed.isnull().sum()
    nan_after = nan_after[nan_after > 0]
    print(f"\nNaN after imputation:\n{nan_after if len(nan_after) > 0 else 'None remaining'}")
    print(f"Output shape: {df_imputed.shape}")

    df_imputed.to_parquet(out_path)
    print(f"Saved: {out_path}")


def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Run the SAITS_MY imputation pipeline.")
    parser.add_argument("--raw", type=pathlib.Path, default=None)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--apply-constraints", action="store_true",
                        help="Apply post-hoc physical-bound constraints to imputed output.")
    parser.add_argument(
        "--sshb",
        action="store_true",
        help="Enable SSHB (mod_p) inference-time refinement on the improved Glocal-IB imputer.",
    )
    parser.add_argument("--use-physical-loss", action="store_true",
                        help="Add the physical-bound auxiliary loss during training.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; reload checkpoint and re-run inference + post-processing.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Master seed for numpy/torch/pygrinder. Default %(default)s.")
    args = parser.parse_args()
    main(
        raw_path=args.raw, out_path=args.out, ckpt_path=args.ckpt,
        apply_constraints=args.apply_constraints,
        use_physical_loss=args.use_physical_loss,
        skip_train=args.skip_train,
        epochs=args.epochs, patience=args.patience,
        n_layers=args.n_layers, d_model=args.d_model,
        seed=args.seed,
        sshb=args.sshb,
    )


if __name__ == "__main__":
    _cli()
