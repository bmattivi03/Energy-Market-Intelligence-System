"""Reproducible post-imputation fixes + train/val/test split builder.

Extracted from notebooks/03_imputation_visualization.ipynb so the split
pipeline is runnable headless. Bounds and the structural-zero indicator name
are imported from src/data/schemas.py (single source of truth).

Usage:
    PYTHONPATH=src python -m preprocessing.build_splits
"""
from __future__ import annotations

import pathlib

import pandas as pd

from data.schemas import (
    SPLIT_BOUNDS,
    STRUCTURAL_ZERO_INDICATOR,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROCESSED = _ROOT / "data" / "processed"
_SPLITS = _ROOT / "data" / "splits"

# Inclusive end timestamps from schemas (hourly resolution); exclusive
# next-period boundary is `end + 1h`, used for `index < boundary` comparisons.
_HOUR = pd.Timedelta("1h")
TRAIN_END = SPLIT_BOUNDS["train"][1] + _HOUR  # exclusive -> 2024-01-01 UTC
VAL_END = SPLIT_BOUNDS["val"][1] + _HOUR      # exclusive -> 2025-01-01 UTC
TEST_END = SPLIT_BOUNDS["test"][1] + _HOUR    # exclusive -> 2025-04-01 UTC

B03_COL = "gen_fossil_coal_gas"
STRUCTURAL_COL = STRUCTURAL_ZERO_INDICATOR


def apply_post_imputation_fixes(
    df_imputed: pd.DataFrame, df_mask: pd.DataFrame
) -> pd.DataFrame:
    """Carbon-ETS ffill+bfill and gen_fossil_coal_gas structural-zero handling."""
    out = df_imputed.copy()

    if "carbon_ets" in out.columns:
        out["carbon_ets"] = out["carbon_ets"].ffill().bfill()

    if B03_COL in df_mask.columns:
        b03_structural = (
            df_mask[B03_COL].reindex(out.index, fill_value=False).astype(bool)
        )
    else:
        b03_structural = pd.Series(False, index=out.index)
    out.loc[b03_structural, B03_COL] = 0.0
    out[STRUCTURAL_COL] = b03_structural.astype(float)
    return out


def split_frame(df: pd.DataFrame):
    """Return (train, val, test) on fixed bounds (2019-2023 / 2024 / Q1-2025)."""
    train = df[df.index < TRAIN_END]
    val = df[(df.index >= TRAIN_END) & (df.index < VAL_END)]
    test = df[(df.index >= VAL_END) & (df.index < TEST_END)]
    return train, val, test


def main() -> None:
    df_imputed = pd.read_parquet(_PROCESSED / "emis_imputed.parquet")
    df_mask = pd.read_parquet(_PROCESSED / "emis_mask.parquet")
    df_clean = apply_post_imputation_fixes(df_imputed, df_mask)
    train, val, test = split_frame(df_clean)

    _SPLITS.mkdir(parents=True, exist_ok=True)
    train.to_parquet(_SPLITS / "train.parquet")
    val.to_parquet(_SPLITS / "val.parquet")
    test.to_parquet(_SPLITS / "test.parquet")

    meta = pd.DataFrame(
        [
            {"split": name, "start": str(part.index.min()),
             "end": str(part.index.max()), "rows": len(part),
             "columns": part.shape[1]}
            for name, part in (("train", train), ("val", val), ("test", test))
        ]
    )
    meta.to_csv(_SPLITS / "split_meta.csv", index=False)
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
