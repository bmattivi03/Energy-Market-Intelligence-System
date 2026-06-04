"""Smoke tests for classical model wrappers - fit + predict + save/load."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module_b.features import build_features, prepare_supervised
from module_b.models import (
    CatBoostQuantileForecaster,
    LightGBMQuantileForecaster,
)


@pytest.fixture
def supervised_dataset(small_df):
    feat = build_features(small_df, bundles=("calendar", "lags"))
    X, y = prepare_supervised(
        feat,
        horizons=range(1, 4),  # h1, h2, h3 - fast
        past_cols=["price_lag1h", "price_lag24h", "price_lag168h", "price_rmean24h"],
        future_cols=["hour_sin", "hour_cos"],
    )
    return X, y


def _expect_quantiles(df: pd.DataFrame, n: int) -> None:
    assert df.shape == (n, 3)
    assert set(df.columns) == {"q10", "q50", "q90"}


def test_lightgbm_fit_predict(supervised_dataset) -> None:
    X, y = supervised_dataset
    m = LightGBMQuantileForecaster(num_boost_round=20, early_stopping_rounds=None)
    m.fit(X, y)
    _expect_quantiles(m.predict_quantiles(X), len(X))


def test_lightgbm_save_load(supervised_dataset, tmp_path) -> None:
    X, y = supervised_dataset
    m = LightGBMQuantileForecaster(num_boost_round=20, early_stopping_rounds=None)
    m.fit(X, y)
    m.save(tmp_path / "lgb")
    m2 = LightGBMQuantileForecaster.load(tmp_path / "lgb")
    np.testing.assert_allclose(m.predict_quantiles(X).to_numpy(), m2.predict_quantiles(X).to_numpy())


def test_catboost_fit_predict(supervised_dataset) -> None:
    X, y = supervised_dataset
    m = CatBoostQuantileForecaster(iterations=30, early_stopping_rounds=10)
    m.fit(X, y)
    _expect_quantiles(m.predict_quantiles(X), len(X))


def test_catboost_save_load(supervised_dataset, tmp_path) -> None:
    X, y = supervised_dataset
    m = CatBoostQuantileForecaster(iterations=30, early_stopping_rounds=10)
    m.fit(X, y)
    m.save(tmp_path / "cb")
    m2 = CatBoostQuantileForecaster.load(tmp_path / "cb")
    np.testing.assert_allclose(m.predict_quantiles(X).to_numpy(), m2.predict_quantiles(X).to_numpy())


def test_catboost_per_quantile_train_calibration(supervised_dataset) -> None:
    """Per-quantile mode (default) fits the upper tail well enough that on the
    training set itself, fewer than 20% of y-values exceed q90. This is a
    sanity check on the structural fix for the MultiQuantile under-fit-tails
    bug (see reports/module_b_catboost_calibration_diagnosis.md).
    """
    X, y = supervised_dataset
    m = CatBoostQuantileForecaster(iterations=200, early_stopping_rounds=None)
    m.fit(X, y)
    q = m.predict_quantiles(X)
    yt = y.to_numpy()
    above_q90 = float(np.mean(yt > q["q90"].to_numpy()))
    below_q10 = float(np.mean(yt < q["q10"].to_numpy()))
    # On train, the model should clearly outperform a misspecified bound.
    assert above_q90 < 0.20, f"y>q90 fraction {above_q90:.3f} too high (per-quantile mode regression?)"
    assert below_q10 < 0.20, f"y<q10 fraction {below_q10:.3f} too high"


def test_catboost_multi_mode_backcompat(supervised_dataset, tmp_path) -> None:
    """Legacy ``mode='multi'`` path still trains and persists correctly."""
    X, y = supervised_dataset
    m = CatBoostQuantileForecaster(iterations=30, early_stopping_rounds=10, mode="multi")
    m.fit(X, y)
    _expect_quantiles(m.predict_quantiles(X), len(X))
    m.save(tmp_path / "cb_multi")
    m2 = CatBoostQuantileForecaster.load(tmp_path / "cb_multi")
    assert m2.mode == "multi"
    np.testing.assert_allclose(
        m.predict_quantiles(X).to_numpy(), m2.predict_quantiles(X).to_numpy(),
    )
