"""Training CLI for Module B (day-ahead price forecasting).

Reproduces the B6 notebook setup that achieved MAE=23.15 on 2025 Q1 test:
- Features built on full dataset (train+val+test) to avoid boundary artifacts
  in rolling stats and spike thresholds
- Curated past/future col selection (matches B6 exactly)
- iterations=500, early_stopping_rounds=30 (B6 values)
- Train on pre-2024, CQR calibrate on 2024 val, evaluate on 2025 Q1 test

Usage
-----
# v1 standalone (no Module A output)
PYTHONPATH=src python -m module_b.train

# v2 with Module A load quantiles
PYTHONPATH=src python -m module_b.train --bundles calendar lags fundamentals spike regime weather load_quantiles

# Load existing checkpoint, skip retraining
PYTHONPATH=src python -m module_b.train --no-fresh
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import ModuleBConfig
from data.loaders import load_test, load_train, load_val
from data.schemas import PRICE_COL
from module_b.evaluation import (
    ConformalQuantileRegressor,
    coverage,
    mae,
    multi_pinball_loss,
    winkler_score,
)
from module_b.features import (
    build_features,
    filter_by_horizon,
    prepare_supervised,
)
from module_b.models import CatBoostQuantileForecaster

# ---------------------------------------------------------------- paths

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
DEFAULT_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "module_b"

PRODUCTION_BUNDLES = list(ModuleBConfig.bundles)
QUANTILES = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------- feature selection (matches B6)

def _select_cols(feat_df: pd.DataFrame, bundles: list[str]) -> tuple[list[str], list[str]]:
    """Curated past/future cols matching the B6 notebook feature selection.

    past_cols: price lags + rolling stats + fundamentals + spike/regime flags
               + load_quantile cols if load_quantiles bundle is active.
    future_cols: subset of calendar + weather aggregates known at target time.
    """
    past_cols = [
        c for c in feat_df.columns if (
            c.startswith("price_lag")
            or c.startswith("price_rmean")
            or c.startswith("price_rstd")
            or c in (
                "residual_load", "renewable_penetration",
                "clean_spark_anchor", "clean_dark_anchor",
                "gas_carbon_interaction",
                "is_high_residual_load", "is_renewable_scarcity",
                "crisis_2022_flag",
            )
            or (c.startswith("load_q") and "load_quantiles" in bundles)
        )
    ]
    future_cols = [
        c for c in (
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos",
            "is_weekend", "is_holiday_DE",
            "weather_mean_wind_speed_100m", "weather_mean_shortwave_radiation",
        )
        if c in feat_df.columns
    ]
    return past_cols, future_cols


# ---------------------------------------------------------------- helpers

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _print_metrics(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    model,
) -> dict:
    q_preds = model.predict_quantiles(X)
    point = q_preds["q50"]
    mae_val = mae(y, point)
    pinball = multi_pinball_loss(y, q_preds, QUANTILES)
    cov = coverage(y, q_preds["q10"], q_preds["q90"])
    winkler = winkler_score(y, q_preds["q10"], q_preds["q90"])
    print(f"  {name:<12} MAE={mae_val:.3f}  pinball={pinball:.3f}  "
          f"coverage={cov:.3f}  winkler={winkler:.1f}")

    X_h, y_h = filter_by_horizon(X, y, range(1, 7))
    if len(y_h):
        q_h = model.predict_quantiles(X_h)
        mae_h = mae(y_h, q_h["q50"])
        pb_h = multi_pinball_loss(y_h, q_h, QUANTILES)
        cov_h = coverage(y_h, q_h["q10"], q_h["q90"])
        print(f"  {'':12} h1-6: MAE={mae_h:.3f}  pinball={pb_h:.3f}  coverage={cov_h:.3f}")

    return {"mae": mae_val, "pinball": pinball, "coverage": cov, "winkler": winkler}


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Module B price forecaster")
    parser.add_argument("--seed",        type=int,   default=ModuleBConfig.seed)
    parser.add_argument("--iterations",  type=int,   default=ModuleBConfig.iterations,
                        help="Max CatBoost iterations (B6 default: 500)")
    parser.add_argument("--early-stopping", type=int, default=ModuleBConfig.early_stopping,
                        help="CatBoost early stopping rounds (B6 default: 30)")
    parser.add_argument("--lr",          type=float, default=ModuleBConfig.lr)
    parser.add_argument(
        "--bundles", nargs="+", default=PRODUCTION_BUNDLES,
    )
    parser.add_argument(
        "--ckpt-dir", type=pathlib.Path, default=DEFAULT_CKPT_DIR,
    )
    parser.add_argument(
        "--fresh", action=argparse.BooleanOptionalAction, default=True,
    )
    args = parser.parse_args()

    set_seed(args.seed)
    ckpt_dir: pathlib.Path = args.ckpt_dir
    catboost_dir = ckpt_dir / "catboost"
    cqr_dir = ckpt_dir / "cqr"

    # ---- 1. load all splits
    print("Loading splits...")
    train_raw = load_train()
    val_raw   = load_val()
    test_raw  = load_test()

    # ---- 2. build features on FULL dataset (avoids rolling/spike boundary artifacts)
    print(f"Building features on full dataset (bundles: {args.bundles})...")
    full_raw  = pd.concat([train_raw, val_raw, test_raw])
    full_feat = build_features(full_raw, args.bundles)

    # Split back by index
    train_feat = full_feat.loc[train_raw.index]
    val_feat   = full_feat.loc[val_raw.index]
    test_feat  = full_feat.loc[test_raw.index]
    print(f"  train: {train_feat.shape}  val: {val_feat.shape}  test: {test_feat.shape}")

    # ---- 3. curated feature selection (matches B6 notebook)
    past_cols, future_cols = _select_cols(full_feat, args.bundles)
    print(f"  past={len(past_cols)} cols, future={len(future_cols)} cols")

    X_train, y_train = prepare_supervised(
        train_feat, past_cols=past_cols, future_cols=future_cols
    )
    X_val, y_val = prepare_supervised(
        val_feat, past_cols=past_cols, future_cols=future_cols
    )
    X_test, y_test = prepare_supervised(
        test_feat, past_cols=past_cols, future_cols=future_cols
    )
    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    # ---- 4. train or load
    if not args.fresh and (catboost_dir / "meta.json").exists():
        print(f"\nLoading checkpoint: {catboost_dir}")
        base_model = CatBoostQuantileForecaster.load(catboost_dir)
    else:
        print(f"\nTraining CatBoostQuantileForecaster "
              f"(iter={args.iterations}, es={args.early_stopping})...")
        base_model = CatBoostQuantileForecaster(
            quantiles=QUANTILES,
            iterations=args.iterations,
            early_stopping_rounds=args.early_stopping,
            learning_rate=args.lr,
            random_state=args.seed,
            mode="per_quantile",
        )
        base_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        print("  Done.")

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        base_model.save(catboost_dir)
        print(f"  Saved: {catboost_dir}")

    # ---- 5. CQR calibration on full val set
    print("\nCalibrating CQR on val set...")
    cqr = ConformalQuantileRegressor(base=base_model, alpha=ModuleBConfig.cqr_alpha)
    cqr.calibrate(X_val, y_val)
    print(f"  delta = {cqr.delta:.4f} EUR/MWh")

    cqr_dir.mkdir(parents=True, exist_ok=True)
    cqr.save(cqr_dir)
    print(f"  Saved: {cqr_dir}")

    # ---- 6. metrics
    print("\n--- Val metrics (raw CatBoost) ---")
    val_raw_m = _print_metrics("val", X_val, y_val, base_model)
    print("\n--- Val metrics (after CQR) ---")
    val_cqr_m = _print_metrics("val+CQR", X_val, y_val, cqr)
    print("\n--- Test metrics (CatBoost + CQR) ---")
    test_m = _print_metrics("test", X_test, y_test, cqr)

    # ---- 7. metadata
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "bundles": list(args.bundles),
        "past_cols": past_cols,
        "future_cols": future_cols,
        "quantiles": list(QUANTILES),
        "seed": args.seed,
        "iterations": args.iterations,
        "early_stopping_rounds": args.early_stopping,
        "learning_rate": args.lr,
        "cqr_alpha": 0.20,
        "cqr_delta": cqr.delta,
        "val_metrics_raw": val_raw_m,
        "val_metrics_cqr": val_cqr_m,
        "test_metrics_cqr": test_m,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    meta_path = ckpt_dir / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved: {meta_path}")


if __name__ == "__main__":
    main()
