"""Apply conformal quantile regression (CQR) to Module A load forecaster.

Calibrates using the first half of the val set (2024 H1).
Reports uncalibrated vs calibrated metrics on val H2 and test.
Saves delta to checkpoints/module_a/cqr_state.pkl.

Usage
-----
PYTHONPATH=src python -m module_a.calibrate
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd

from data.loaders import load_val, load_test
from module_a.evaluation import full_report
from module_a.features import ALL_BUNDLES, TARGET_COL, build_features
from module_a.model import MultiScaleLSTMForecaster

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
CKPT      = PROJECT_ROOT / "checkpoints" / "module_a" / "best.pt"
CQR_STATE = PROJECT_ROOT / "checkpoints" / "module_a" / "cqr_state.pkl"

ALPHA = 0.20  # target 80% coverage interval


def _align(long_preds: pd.DataFrame, actual: pd.Series) -> pd.DataFrame:
    rows = []
    for (origin, h), row in long_preds.iterrows():
        target_ts = origin + pd.Timedelta(hours=int(h))
        if target_ts in actual.index:
            rows.append({"q10": row["q10"], "q50": row["q50"],
                         "q90": row["q90"], "y": actual[target_ts]})
    return pd.DataFrame(rows)


def compute_delta(long_preds: pd.DataFrame, actual: pd.Series) -> float:
    """Barber finite-sample CQR delta for 80% interval."""
    df = _align(long_preds, actual)
    scores = np.maximum(
        df["q10"].values - df["y"].values,
        df["y"].values  - df["q90"].values,
    )
    n = len(scores)
    level = np.ceil((n + 1) * (1 - ALPHA)) / n
    return float(np.quantile(scores, np.clip(level, 0.0, 1.0)))


def apply_cqr(long_preds: pd.DataFrame, delta: float) -> pd.DataFrame:
    out = long_preds.copy()
    out["q10"] = out["q10"] - delta
    out["q90"] = out["q90"] + delta
    return out


def main() -> None:
    print(f"Loading checkpoint: {CKPT}")
    forecaster = MultiScaleLSTMForecaster.load(CKPT)

    val_raw  = load_val()
    test_raw = load_test()

    val_feat  = build_features(val_raw,  bundles=list(ALL_BUNDLES))
    test_feat = build_features(test_raw, bundles=list(ALL_BUNDLES))

    val_actual  = val_raw[TARGET_COL]
    test_actual = test_raw[TARGET_COL]

    # Split val in half: first half → calibration, second half → held-out report
    mid = len(val_raw) // 2
    cal_idx  = val_raw.index[:mid]
    val2_idx = val_raw.index[mid:]

    cal_feat  = val_feat.loc[cal_idx]
    val2_feat = val_feat.loc[val2_idx]
    cal_actual  = val_actual.loc[cal_idx]
    val2_actual = val_actual.loc[val2_idx]

    print(f"Calibration : {cal_idx[0]}  –  {cal_idx[-1]}  ({len(cal_idx)} h)")
    print(f"Val held-out: {val2_idx[0]}  –  {val2_idx[-1]}  ({len(val2_idx)} h)")

    print("\nPredicting on calibration set...")
    cal_preds = forecaster.predict_quantiles(cal_feat, stride=1)
    delta = compute_delta(cal_preds, cal_actual)
    print(f"CQR delta = {delta:+.1f} MW  (alpha={ALPHA}, target coverage 80%)")

    CQR_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(CQR_STATE, "wb") as f:
        pickle.dump({"alpha": ALPHA, "delta": delta}, f)
    print(f"Saved: {CQR_STATE}")

    print("\nPredicting on val held-out...")
    val2_preds     = forecaster.predict_quantiles(val2_feat, stride=1)
    val2_preds_cal = apply_cqr(val2_preds, delta)

    full_report(val2_preds,     val2_actual, split_name="val2 (uncalibrated)")
    full_report(val2_preds_cal, val2_actual, split_name="val2 (CQR-calibrated)")

    print("\nPredicting on test set...")
    test_preds     = forecaster.predict_quantiles(test_feat, stride=1)
    test_preds_cal = apply_cqr(test_preds, delta)

    full_report(test_preds,     test_actual, split_name="test (uncalibrated)")
    full_report(test_preds_cal, test_actual, split_name="test (CQR-calibrated)")


if __name__ == "__main__":
    main()
