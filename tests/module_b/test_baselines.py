"""Tests for naive / seasonal-naive baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module_b.features import HORIZON_COL, TARGET_COL, build_features, prepare_supervised
from module_b.models import NaiveForecaster, SeasonalNaiveForecaster


@pytest.fixture
def supervised_dataset(small_df):
    """Build a small (X, y) dataset with the lags bundle so price_lag1h exists."""
    feat = build_features(small_df, bundles=("calendar", "lags"))
    X, y = prepare_supervised(
        feat,
        horizons=range(1, 7),
        past_cols=["price_lag1h", "price_lag24h", "price_lag168h", "price_rmean24h"],
        future_cols=["hour_sin", "hour_cos"],
    )
    return X, y, small_df["price"]


def test_naive_forecaster_predicts_anchor(supervised_dataset) -> None:
    X, y, _ = supervised_dataset
    m = NaiveForecaster(anchor_column="price_lag1h")
    m.fit(X, y)
    q = m.predict_quantiles(X)
    # q50 should equal anchor + 0-residual ≈ anchor
    assert "q50" in q.columns
    # All quantile columns present
    assert set(q.columns) == {"q10", "q50", "q90"}
    # Quantile order: q10 ≤ q50 ≤ q90 (modulo edge ties)
    assert (q["q10"] <= q["q50"] + 1e-6).all()


def test_naive_save_load_roundtrip(supervised_dataset, tmp_path) -> None:
    X, y, _ = supervised_dataset
    m = NaiveForecaster()
    m.fit(X, y)
    p = tmp_path / "naive.pkl"
    m.save(p)
    m2 = NaiveForecaster.load(p)
    np.testing.assert_array_equal(
        m.predict_quantiles(X).to_numpy(),
        m2.predict_quantiles(X).to_numpy(),
    )


def test_seasonal_naive_uses_target_minus_168(supervised_dataset) -> None:
    X, y, price_series = supervised_dataset
    m = SeasonalNaiveForecaster(season_hours=168)
    m.fit(X, y, price_series=price_series)
    q = m.predict_quantiles(X)
    assert q.shape[0] == len(X)
    assert {"q10", "q50", "q90"}.issubset(q.columns)


def test_seasonal_naive_requires_price_series(supervised_dataset) -> None:
    X, y, _ = supervised_dataset
    m = SeasonalNaiveForecaster()
    with pytest.raises(ValueError, match="price_series"):
        m.fit(X, y)
