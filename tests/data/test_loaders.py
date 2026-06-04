"""Unit tests for src/data/loaders.py and src/data/schemas.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import schemas
from data.loaders import load_split


def test_schema_constants_are_distinct() -> None:
    # No accidental overlaps between feature groups
    assert set(schemas.GENERATION_COLS).isdisjoint(set(schemas.WEATHER_COLS))
    assert set(schemas.GENERATION_COLS).isdisjoint(set(schemas.FUEL_COLS))
    assert set(schemas.WEATHER_COLS).isdisjoint(set(schemas.FUEL_COLS))


def test_weather_cols_naming_convention() -> None:
    # Every weather column is "<city>_<variable>"
    for col in schemas.WEATHER_COLS:
        city, _sep, var = col.partition("_")
        assert city in schemas.CITY_NAMES, col
        assert any(var.startswith(v.split("_")[0]) for v in schemas.WEATHER_VARIABLES), col


def test_split_bounds_are_contiguous() -> None:
    # Train ends right before val starts; val ends right before test starts.
    bounds = schemas.SPLIT_BOUNDS
    assert bounds["train"][1] < bounds["val"][0]
    assert bounds["val"][1] < bounds["test"][0]


@pytest.mark.skipif(
    not (schemas := __import__("data.schemas", fromlist=["SPLIT_BOUNDS"])),
    reason="splits parquets must exist for this test"
)
def test_load_split_train_returns_canonical_columns() -> None:
    df = load_split("train")
    assert schemas.PRICE_COL in df.columns
    assert schemas.LOAD_COL in df.columns
    for col in schemas.GENERATION_COLS:
        assert col in df.columns, f"missing {col}"
    assert df.index.tz is not None


def test_load_split_with_mask_indicators() -> None:
    df = load_split("train", with_mask_indicators=True)
    indicators = [c for c in df.columns if c.endswith("_was_missing")]
    assert len(indicators) > 0
    for c in indicators:
        assert df[c].dtype == np.float32
        # 0 or 1 values only
        unique = set(df[c].unique())
        assert unique.issubset({0.0, 1.0}), f"{c} has non-binary values: {unique}"
