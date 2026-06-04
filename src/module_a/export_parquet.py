"""Export CQR-calibrated load quantile predictions to parquet.

Loads best.pt + cqr_state.pkl, runs predict_quantiles on train/val/test,
applies CQR adjustment to q10/q90, saves wide-format parquet consumed by
module_b.features (load_quantiles bundle).

Usage
-----
PYTHONPATH=src python -m module_a.export_parquet
"""

from __future__ import annotations

import pathlib
import pickle

import pandas as pd

from data.loaders import load_test, load_train, load_val
from module_a.features import ALL_BUNDLES, build_features
from module_a.model import MultiScaleLSTMForecaster
from module_a.train import OUTPUT_DIR, OUTPUT_PARQUET, _to_wide

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
CKPT      = PROJECT_ROOT / "checkpoints" / "module_a" / "best.pt"
CQR_STATE = PROJECT_ROOT / "checkpoints" / "module_a" / "cqr_state.pkl"


def main() -> None:
    print(f"Loading checkpoint: {CKPT}")
    forecaster = MultiScaleLSTMForecaster.load(CKPT)

    with open(CQR_STATE, "rb") as f:
        cqr = pickle.load(f)
    delta = cqr["delta"]
    print(f"CQR delta = {delta:+.1f} MW")

    train_raw = load_train()
    val_raw   = load_val()
    test_raw  = load_test()

    bundles = list(ALL_BUNDLES)

    longs = []
    for name, raw in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        print(f"Predicting {name}...")
        feat = build_features(raw, bundles=bundles)
        long = forecaster.predict_quantiles(feat, stride=1)
        long["q10"] = long["q10"] - delta
        long["q90"] = long["q90"] + delta
        longs.append(long)

    all_long = pd.concat(longs).sort_index()
    wide = _to_wide(all_long)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUTPUT_PARQUET)
    print(f"\nSaved: {OUTPUT_PARQUET}  ({wide.shape[0]:,} rows × {wide.shape[1]} cols)")
    print("Columns:", list(wide.columns[:6]), "...")


if __name__ == "__main__":
    main()
