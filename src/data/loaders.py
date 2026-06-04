"""Mask-aware loaders for raw, imputed, mask, and split parquets.

Conventions:
* All returned DataFrames have a tz-aware (UTC) DatetimeIndex.
* ``load_split`` returns the canonical pre-built train/val/test parquets used
  by the ML modules. ``with_mask_indicators=True`` adds per-column
  ``*_was_missing`` indicators derived from the mask parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

SplitName = Literal["train", "val", "test"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"


def _resolve(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw NaN-bearing parquet."""
    return pd.read_parquet(_resolve(path, _DATA_DIR / "processed" / "emis_raw.parquet"))


def load_imputed(path: str | Path | None = None) -> pd.DataFrame:
    """Load the post-imputation parquet."""
    return pd.read_parquet(_resolve(path, _DATA_DIR / "processed" / "emis_imputed.parquet"))


def load_mask(path: str | Path | None = None) -> pd.DataFrame:
    """Load the bool/uint8 mask parquet (1 = cell was imputed)."""
    return pd.read_parquet(_resolve(path, _DATA_DIR / "processed" / "emis_mask.parquet"))


def load_split(
    split: SplitName,
    *,
    path: str | Path | None = None,
    with_mask_indicators: bool = False,
    mask_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load a pre-built split parquet.

    Parameters
    ----------
    split : one of ``"train"``, ``"val"``, ``"test"``.
    path : optional override for the splits directory.
    with_mask_indicators : if True, joins ``mask`` parquet and appends
        ``<col>_was_missing`` Float32 columns for each requested column.
    mask_columns : columns for which to add indicators (default: all columns
        that have at least one mask entry).
    """
    base = _resolve(path, _DATA_DIR / "splits") / f"{split}.parquet"
    df = pd.read_parquet(base)
    if not with_mask_indicators:
        return df
    mask = load_mask()
    mask = mask.loc[mask.index.intersection(df.index)]
    if mask_columns is None:
        mask_columns = [c for c in mask.columns if mask[c].sum() > 0]
    out = df.copy()
    for col in mask_columns:
        if col not in mask.columns:
            continue
        indicator = mask[col].reindex(out.index, fill_value=0).astype(np.float32)
        out[f"{col}_was_missing"] = indicator
    return out


# Convenience wrappers used by module_a.train and notebooks.
def load_train(**kwargs) -> pd.DataFrame:
    return load_split("train", **kwargs)


def load_val(**kwargs) -> pd.DataFrame:
    return load_split("val", **kwargs)


def load_test(**kwargs) -> pd.DataFrame:
    return load_split("test", **kwargs)
