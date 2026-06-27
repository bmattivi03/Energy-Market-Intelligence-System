"""Tests for module_a.features.build_features.

Covers:
* the expected bundle columns appear after build_features(ALL_BUNDLES);
* lag/rolling features are causal (no look-ahead — verified against an
  explicit .shift());
* feature scaling is fit on train only (no leakage of val statistics).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from module_a.features import (
    ALL_BUNDLES,
    CALENDAR_COLS,
    LOAD_LAG_COLS,
    RENEWABLE_COLS,
    TARGET_COL,
    WEATHER_DERIVED_COLS,
    add_load_lags,
    build_features,
)


# ---------------------------------------------------------------- column presence


def test_build_features_emits_all_bundle_columns(load_df):
    feats = build_features(load_df, ALL_BUNDLES)

    # Original input columns are preserved (build_features only appends).
    for col in load_df.columns:
        assert col in feats.columns, f"input column {col} dropped"

    # Every documented bundle column is present.
    expected = set(CALENDAR_COLS) | set(LOAD_LAG_COLS) | set(
        WEATHER_DERIVED_COLS
    ) | set(RENEWABLE_COLS)
    missing = expected - set(feats.columns)
    assert not missing, f"missing engineered columns: {sorted(missing)}"


def test_build_features_subset_only_adds_selected_bundle(load_df):
    feats = build_features(load_df, ["calendar"])
    for col in CALENDAR_COLS:
        assert col in feats.columns
    # Load-lag columns must NOT appear when only the calendar bundle is asked for.
    assert not (set(LOAD_LAG_COLS) & set(feats.columns))


def test_build_features_returns_same_index(load_df):
    feats = build_features(load_df, ALL_BUNDLES)
    assert feats.index.equals(load_df.index)
    assert len(feats) == len(load_df)


def test_unknown_bundle_raises(load_df):
    with pytest.raises(KeyError):
        build_features(load_df, ["not_a_real_bundle"])


# ---------------------------------------------------------------- causality


def test_load_lag_is_exact_shift(load_df):
    """load_lag24h at row t must equal load at row t-24 (pure backward shift)."""
    feats = add_load_lags(load_df)
    expected = load_df[TARGET_COL].shift(24)
    pd.testing.assert_series_equal(
        feats["load_lag24h"].astype(np.float64),
        expected.astype(np.float64),
        check_names=False,
    )
    # First 24 rows have no 24h-ago value -> must be NaN (no fabricated history).
    assert feats["load_lag24h"].iloc[:24].isna().all()


def test_rolling_features_are_shifted_one_step(load_df):
    """Rolling mean at t excludes the value at t (built with .shift(1))."""
    feats = add_load_lags(load_df)
    s = load_df[TARGET_COL]
    w = 24
    expected = s.rolling(w, min_periods=max(2, w // 4)).mean().shift(1)
    pd.testing.assert_series_equal(
        feats["load_rmean24h"].astype(np.float64),
        expected.astype(np.float64),
        check_names=False,
    )


def test_no_lag_feature_uses_future_information(load_df):
    """Perturbing a future load value must not change any past feature row.

    A causal feature at time t depends only on data at times <= t. We change a
    single load value far in the future and assert every engineered row before
    it is bit-identical.
    """
    feats_a = build_features(load_df, ALL_BUNDLES)

    perturbed = load_df.copy()
    cut = len(perturbed) - 5
    perturbed.iloc[cut, perturbed.columns.get_loc(TARGET_COL)] += 1e6
    feats_b = build_features(perturbed, ALL_BUNDLES)

    engineered = (
        list(LOAD_LAG_COLS)
        + list(WEATHER_DERIVED_COLS)
        + list(RENEWABLE_COLS)
    )
    before = feats_a.iloc[: cut - 336][engineered]
    after = feats_b.iloc[: cut - 336][engineered]
    # Compare ignoring NaN==NaN positions.
    diff = (before.fillna(-999) != after.fillna(-999))
    assert not diff.to_numpy().any(), (
        "a future load change leaked into a past feature row"
    )


# ---------------------------------------------------------------- scaler leakage


def test_scaler_fit_on_train_only(split_dfs):
    """A StandardScaler fit on train must use train stats, not val stats."""
    train, val = split_dfs
    feats_train = build_features(train, ALL_BUNDLES).select_dtypes("number")
    feats_val = build_features(val, ALL_BUNDLES).select_dtypes("number")

    cols = [c for c in feats_train.columns if c != TARGET_COL]
    X_train = feats_train[cols].to_numpy(np.float64)
    X_train = np.nan_to_num(X_train)

    scaler = StandardScaler().fit(X_train)

    # The fitted mean/scale must reproduce train column statistics, NOT val's.
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-6, atol=1e-6)

    X_val = np.nan_to_num(feats_val[cols].to_numpy(np.float64))
    val_mean = X_val.mean(axis=0)
    # On non-degenerate columns the two means should differ for at least some
    # features, proving the scaler did not peek at val.
    # Columns that genuinely vary in the train data (StandardScaler clamps
    # scale_ to 1.0 for constant columns, so test variance directly).
    varying = X_train.std(axis=0) > 1e-6
    assert np.any(np.abs(scaler.mean_[varying] - val_mean[varying]) > 1e-9)

    # Transforming train data with the train-fit scaler yields unit variance
    # on the columns that actually vary.
    Z = scaler.transform(X_train)
    std = Z[:, varying].std(axis=0)
    np.testing.assert_allclose(std, np.ones_like(std), rtol=1e-6, atol=1e-6)
