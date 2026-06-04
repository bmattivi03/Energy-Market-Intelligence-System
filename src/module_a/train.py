"""Training and inference CLI for Module A (load forecasting).

Usage examples
--------------
# Full train from scratch
PYTHONPATH=src python -m module_a.train --epochs 200 --patience 20

# Skip training, load existing checkpoint and re-export parquet
PYTHONPATH=src python -m module_a.train --fresh=false

# Quick smoke test
PYTHONPATH=src python -m module_a.train --epochs 5 --patience 999
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from config import ModuleAConfig
from data.loaders import load_test, load_train, load_val
from module_a.features import ALL_BUNDLES, TARGET_COL, build_features
from module_a.model import HORIZON, QUANTILES, MultiScaleLSTMForecaster
from utils.reproducibility import set_seed

# ---------------------------------------------------------------- paths

PROJECT_ROOT  = pathlib.Path(__file__).parents[2]
CKPT_PATH     = PROJECT_ROOT / "checkpoints" / "module_a" / "best.pt"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "module_a"
OUTPUT_PARQUET = OUTPUT_DIR / "load_quantiles.parquet"


# ---------------------------------------------------------------- helpers

def _to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (origin_ts, horizon_h) long form → wide form.

    Output index : origin_ts
    Output cols  : load_q{10,50,90}_h{1..24}  (72 columns)
    """
    wide = long_df.unstack("horizon_h")
    wide.columns = [
        f"load_{col}_h{h}"
        for col, h in wide.columns
    ]
    wide.index.name = "datetime_utc"
    return wide


def _eval_metrics(
    long_preds: pd.DataFrame,
    split_df: pd.DataFrame,
    split_name: str,
) -> None:
    """Print pinball loss and q50 MAE/RMSE for one split."""
    actual_load = split_df[TARGET_COL]

    pinball_vals: dict[str, float] = {}
    errors_q50: list[float] = []

    for (origin, h), row in long_preds.iterrows():
        target_ts = origin + pd.Timedelta(hours=int(h))
        if target_ts not in actual_load.index:
            continue
        actual = actual_load[target_ts]
        for qi, q in enumerate(QUANTILES):
            col = f"q{int(round(q * 100))}"
            pred = row[col]
            err = actual - pred
            pb = q * err if err >= 0 else (q - 1) * err
            key = col
            pinball_vals[key] = pinball_vals.get(key, 0.0) + pb
            if col == "q50":
                errors_q50.append(actual - pred)

    n = len(long_preds)
    print(f"\n{split_name} metrics  (n={n:,} origin-horizon pairs)")
    for col in ["q10", "q50", "q90"]:
        if col in pinball_vals:
            print(f"  pinball {col}: {pinball_vals[col] / n:.2f} MW")

    if errors_q50:
        arr = np.array(errors_q50)
        mae  = np.abs(arr).mean()
        rmse = np.sqrt((arr ** 2).mean())
        print(f"  q50 MAE : {mae:.1f} MW")
        print(f"  q50 RMSE: {rmse:.1f} MW")


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Module A load forecaster")
    parser.add_argument("--seed",       type=int,   default=ModuleAConfig.seed)
    parser.add_argument("--epochs",     type=int,   default=ModuleAConfig.epochs)
    parser.add_argument("--patience",   type=int,   default=ModuleAConfig.patience)
    parser.add_argument("--batch-size", type=int,   default=ModuleAConfig.batch_size)
    parser.add_argument("--lr",         type=float, default=ModuleAConfig.lr)
    parser.add_argument("--hidden",            type=int,   default=ModuleAConfig.hidden)
    parser.add_argument("--dropout",           type=float, default=ModuleAConfig.dropout)
    parser.add_argument("--num-layers-short",  type=int,   default=ModuleAConfig.num_layers_short)
    parser.add_argument("--num-layers-long",   type=int,   default=ModuleAConfig.num_layers_long)
    parser.add_argument(
        "--bundles", nargs="+", default=list(ALL_BUNDLES),
        help="Feature bundles to use (default: all)",
    )
    parser.add_argument(
        "--fresh", action=argparse.BooleanOptionalAction, default=True,
        help="Retrain from scratch even if checkpoint exists (default: True)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- 1. load splits
    print("Loading splits...")
    train_raw = load_train()
    val_raw   = load_val()
    test_raw  = load_test()

    # ---- 2. build features
    print(f"Building features (bundles: {args.bundles})...")
    train_feat = build_features(train_raw, bundles=args.bundles)
    val_feat   = build_features(val_raw,   bundles=args.bundles)
    test_feat  = build_features(test_raw,  bundles=args.bundles)
    print(f"  train: {train_feat.shape}  val: {val_feat.shape}  test: {test_feat.shape}")

    # ---- 3. fit or load
    forecaster = MultiScaleLSTMForecaster(
        hidden=args.hidden,
        dropout=args.dropout,
        num_layers_short=args.num_layers_short,
        num_layers_long=args.num_layers_long,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        random_state=args.seed,
        checkpoint_path=CKPT_PATH,
    )

    if not args.fresh and CKPT_PATH.exists():
        print(f"Loading checkpoint: {CKPT_PATH}")
        forecaster = MultiScaleLSTMForecaster.load(CKPT_PATH)
    else:
        print("Training...")
        forecaster.fit(train_feat, val_feat_df=val_feat)

    # ---- 4. predict all three splits
    print("\nGenerating quantile forecasts...")
    train_long = forecaster.predict_quantiles(train_feat, stride=1)
    val_long   = forecaster.predict_quantiles(val_feat,   stride=1)
    test_long  = forecaster.predict_quantiles(test_feat,  stride=1)

    # ---- 5. evaluation
    _eval_metrics(train_long, train_raw, "train")
    _eval_metrics(val_long,   val_raw,   "val")
    _eval_metrics(test_long,  test_raw,  "test")

    # ---- 6. export wide parquet (consumed by module_b.features.add_load_quantiles)
    all_long = pd.concat([train_long, val_long, test_long]).sort_index()
    wide = _to_wide(all_long)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUTPUT_PARQUET)
    print(f"\nSaved: {OUTPUT_PARQUET}  ({wide.shape[0]:,} rows x {wide.shape[1]} cols)")
    print("Columns:", list(wide.columns[:6]), "...")


if __name__ == "__main__":
    main()
