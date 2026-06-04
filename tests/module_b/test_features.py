"""Unit tests for src/module_b/features/*.py and the dataset reshaper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module_b.features import (
    CALENDAR_COLS,
    HORIZON_COL,
    ORIGIN_COL,
    REGISTRY,
    TARGET_COL,
    add_calendar,
    add_fundamentals,
    add_price_lags,
    add_regime,
    add_spike,
    add_weather,
    build_features,
    prepare_supervised,
    weather_columns,
)


def test_add_calendar_attaches_all_columns(small_df: pd.DataFrame) -> None:
    out = add_calendar(small_df)
    for c in CALENDAR_COLS:
        assert c in out.columns


def test_add_calendar_holiday_flag_for_new_years_day(small_df: pd.DataFrame) -> None:
    out = add_calendar(small_df)
    new_years = out.loc[out.index.normalize() == pd.Timestamp("2023-01-01", tz="UTC")]
    assert (new_years["is_holiday_DE"] == 1.0).all()


def test_add_calendar_weekend_flag(small_df: pd.DataFrame) -> None:
    out = add_calendar(small_df)
    saturday = out.loc[out.index.dayofweek == 5]
    assert (saturday["is_weekend"] == 1.0).all()


def test_add_price_lags_creates_lag_24(small_df: pd.DataFrame) -> None:
    out = add_price_lags(small_df, lags=(24,), windows=())
    assert "price_lag24h" in out.columns
    # First 24 values are NaN
    assert out["price_lag24h"].iloc[:24].isna().all()
    # Rest equal to price[t-24]
    assert np.allclose(out["price_lag24h"].iloc[24:].values, small_df["price"].iloc[:-24].values)


def test_add_fundamentals_residual_load_is_not_net_load(small_df: pd.DataFrame) -> None:
    out = add_fundamentals(small_df)
    assert "residual_load" in out.columns
    # Sanity: residual = load − wind − solar − ror, must NOT equal total - wind - solar
    expected = (
        small_df["load"]
        - small_df["gen_wind_onshore"]
        - small_df["gen_wind_offshore"]
        - small_df["gen_solar"]
        - small_df["gen_hydro_ror"]
    )
    assert np.allclose(out["residual_load"].values, expected.values)


def test_add_fundamentals_clean_spreads_present(small_df: pd.DataFrame) -> None:
    out = add_fundamentals(small_df)
    for c in ("clean_spark_anchor", "clean_dark_anchor", "gas_carbon_interaction"):
        assert c in out.columns


def test_add_spike_runs_after_fundamentals(small_df: pd.DataFrame) -> None:
    feat = add_fundamentals(small_df)
    out = add_spike(feat)
    assert "is_high_residual_load" in out.columns
    assert "is_renewable_scarcity" in out.columns


def test_add_regime_crisis_flag(small_df: pd.DataFrame) -> None:
    # small_df is in 2023 - entirely inside the crisis window
    out = add_regime(small_df)
    assert (out["crisis_2022_flag"] == 1.0).all()


def test_add_weather_creates_lags_and_aggregates(small_df: pd.DataFrame) -> None:
    out = add_weather(small_df)
    assert "berlin_temperature_2m_lag1" in out.columns
    assert "weather_mean_temperature_2m" in out.columns


def test_weather_columns_excludes_non_weather() -> None:
    cols = weather_columns()
    for c in cols:
        assert any(v in c for v in ("temperature_2m", "wind_speed", "shortwave_radiation"))


def test_registry_resolves_dependencies(small_df: pd.DataFrame) -> None:
    # spike requires fundamentals → registry must add fundamentals first
    order = REGISTRY.resolve_order(("spike",))
    # look up each spec's key from the registry (name field was removed)
    spec_to_key = {v: k for k, v in REGISTRY.items()}
    names = [spec_to_key[b] for b in order]
    assert names == ["fundamentals", "spike"]


def test_registry_handles_unknown_bundle() -> None:
    with pytest.raises(KeyError, match="Unknown bundle"):
        REGISTRY.resolve_order(("nonexistent",))


def test_build_features_full_default_bundle(small_df: pd.DataFrame) -> None:
    out = build_features(
        small_df,
        bundles=("calendar", "lags", "fundamentals", "spike", "regime", "weather"),
    )
    expected = {
        "hour_sin", "is_holiday_DE", "price_lag24h", "residual_load",
        "is_renewable_scarcity", "crisis_2022_flag", "weather_mean_wind_speed_10m",
    }
    missing = expected - set(out.columns)
    assert not missing, f"missing features: {missing}"


def test_prepare_supervised_layout(small_df: pd.DataFrame) -> None:
    feat = build_features(small_df, bundles=("calendar", "lags"))
    past_cols = ["price_lag24h", "price_rmean24h"]
    future_cols = ["hour_sin", "hour_cos"]
    X, y = prepare_supervised(
        feat,
        horizons=range(1, 7),
        past_cols=past_cols,
        future_cols=future_cols,
    )
    assert HORIZON_COL in X.columns
    assert ORIGIN_COL in X.columns
    assert TARGET_COL in X.columns
    assert "fut_hour_sin" in X.columns  # future cols get prefixed
    # Per-horizon row count is roughly equal
    counts = X[HORIZON_COL].value_counts().sort_index()
    assert (counts.values > 0).all()
    # Index alignment
    assert len(X) == len(y)


def test_prepare_supervised_target_is_price_at_t_plus_h(small_df: pd.DataFrame) -> None:
    feat = build_features(small_df, bundles=("calendar", "lags"))
    X, y = prepare_supervised(
        feat,
        horizons=[1],
        past_cols=["price_lag24h"],
        future_cols=[],
    )
    # For horizon 1, y[i] should equal price at origin_ts[i] + 1h
    sample = X.iloc[0]
    target_ts = sample[TARGET_COL]
    assert np.isclose(y.iloc[0], small_df.loc[target_ts, "price"])
