"""Unit tests for src/preprocessing/physical_constraints.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.physical_constraints import (
    CARBON_ETS_INCEPTION,
    CARBON_ETS_PRE_LAUNCH_INDICATOR,
    STRUCTURAL_ZERO_INDICATOR,
    ConstraintConfig,
    add_structural_zero_indicator,
    apply_all,
    clip_generation,
    enforce_flow_sign_coherence,
    enforce_solar_night_zero,
    restore_carbon_ets_pre_launch,
)


@pytest.fixture
def synthetic_index() -> pd.DatetimeIndex:
    return pd.date_range("2019-06-01", periods=24 * 7, freq="h", tz="UTC")


def test_clip_generation_removes_negatives() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "gen_solar": [-5.0, 100, -1.0, 50],
            "gen_wind_onshore": [-200, 1000, 2000, -1],
            "gen_hydro_pumped": [-500, 200, -300, 100],  # signed: must NOT be clipped
        },
        index=idx,
    )
    out = clip_generation(df)
    assert (out["gen_solar"] >= 0).all()
    assert (out["gen_wind_onshore"] >= 0).all()
    # Pumped storage preserves its negative values
    assert (out["gen_hydro_pumped"] == df["gen_hydro_pumped"]).all()
    # Original df is unmodified
    assert df["gen_solar"].iloc[0] == -5.0


def test_enforce_solar_night_zero_zeroes_at_midnight() -> None:
    # June 1 in Germany — midnight UTC = local 02:00, sun well below horizon
    idx = pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"gen_solar": np.full(24, 1000.0)}, index=idx)
    out = enforce_solar_night_zero(df)
    # At UTC 0, 1, 2, 3, 22, 23 (i.e. local night for German summer), expect zero
    assert out["gen_solar"].iloc[0] == 0.0
    assert out["gen_solar"].iloc[23] == 0.0
    # At UTC 12 (local noon), expect ~unchanged
    assert out["gen_solar"].iloc[12] == pytest.approx(1000.0, rel=1e-6)


def test_enforce_solar_night_zero_handles_missing_column() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({"other": [1, 2, 3, 4]}, index=idx)
    out = enforce_solar_night_zero(df)  # no gen_solar; should be no-op
    pd.testing.assert_frame_equal(out, df)


def test_enforce_flow_sign_coherence_resolves_both_positive() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "FR_to_DELU": [500.0, 0, 1000, 0],
            "DELU_to_FR": [200.0, 100, 1500, 0],
            "AT_to_DELU": [0.0, 0, 0, 0],
            "DELU_to_AT": [0.0, 0, 0, 0],
            "CH_to_DELU": [0.0, 0, 0, 0],
            "DELU_to_CH": [0.0, 0, 0, 0],
        },
        index=idx,
    )
    out = enforce_flow_sign_coherence(df)
    # row 0: FR_to_DELU=500 wins over DELU_to_FR=200 → DELU_to_FR=0, FR_to_DELU=500
    assert out["FR_to_DELU"].iloc[0] == 500.0
    assert out["DELU_to_FR"].iloc[0] == 0.0
    # row 2: DELU_to_FR=1500 wins over FR_to_DELU=1000 → FR_to_DELU=0
    assert out["DELU_to_FR"].iloc[2] == 1500.0
    assert out["FR_to_DELU"].iloc[2] == 0.0
    # rows 1, 3: untouched (no double-positive)
    assert out["FR_to_DELU"].iloc[1] == 0.0
    assert out["DELU_to_FR"].iloc[1] == 100.0


def test_restore_carbon_ets_pre_launch() -> None:
    idx = pd.date_range("2021-09-01", periods=24 * 60, freq="h", tz="UTC")
    df = pd.DataFrame({"carbon_ets": np.full(len(idx), 50.0)}, index=idx)
    out = restore_carbon_ets_pre_launch(df)
    assert CARBON_ETS_PRE_LAUNCH_INDICATOR in out.columns
    # Pre-launch should be NaN
    pre = out.index < CARBON_ETS_INCEPTION
    assert out.loc[pre, "carbon_ets"].isna().all()
    assert (out.loc[pre, CARBON_ETS_PRE_LAUNCH_INDICATOR] == 1.0).all()
    # Post-launch should be unchanged + indicator = 0
    post = out.index >= CARBON_ETS_INCEPTION
    assert (out.loc[post, "carbon_ets"] == 50.0).all()
    assert (out.loc[post, CARBON_ETS_PRE_LAUNCH_INDICATOR] == 0.0).all()


def test_add_structural_zero_indicator_matches_raw_nan_pattern() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    raw = pd.DataFrame(
        {"gen_fossil_coal_gas": [np.nan, 100, np.nan, np.nan, 200, 300, np.nan, np.nan, 400, np.nan]},
        index=idx,
    )
    imputed = pd.DataFrame({"gen_fossil_coal_gas": np.zeros(10)}, index=idx)
    out = add_structural_zero_indicator(imputed, raw)
    expected = raw["gen_fossil_coal_gas"].isna().astype(np.float32)
    pd.testing.assert_series_equal(
        out[STRUCTURAL_ZERO_INDICATOR], expected, check_names=False
    )


def test_apply_all_runs_end_to_end() -> None:
    idx = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    raw = pd.DataFrame(
        {
            "gen_solar": np.full(24, np.nan),
            "gen_fossil_coal_gas": np.full(24, np.nan),
            "FR_to_DELU": np.zeros(24),
            "DELU_to_FR": np.zeros(24),
            "carbon_ets": np.full(24, np.nan),
        },
        index=idx,
    )
    imputed = pd.DataFrame(
        {
            "gen_solar": np.full(24, 500.0),  # positive at night → must zero out
            "gen_fossil_coal_gas": np.full(24, 0.0),  # structural zero
            "FR_to_DELU": np.full(24, 100.0),  # both positive
            "DELU_to_FR": np.full(24, 200.0),
            "carbon_ets": np.full(24, 50.0),  # all pre-launch → must NaN
        },
        index=idx,
    )
    out = apply_all(imputed, raw_df=raw)
    # 1. Solar zeroed at night
    assert out["gen_solar"].iloc[0] == 0.0
    # 2. Coal/gas structural zero indicator added
    assert STRUCTURAL_ZERO_INDICATOR in out.columns
    assert (out[STRUCTURAL_ZERO_INDICATOR] == 1.0).all()
    # 3. Both-positive flows resolved
    assert (out["FR_to_DELU"] == 0.0).all()  # smaller of the two
    assert (out["DELU_to_FR"] == 200.0).all()
    # 4. Pre-launch carbon NaN
    assert out["carbon_ets"].isna().all()
    assert (out[CARBON_ETS_PRE_LAUNCH_INDICATOR] == 1.0).all()


def test_apply_all_requires_raw_for_structural_zero() -> None:
    df = pd.DataFrame(
        {"gen_solar": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    )
    with pytest.raises(ValueError, match="add_structural_zero requires raw_df"):
        apply_all(df, raw_df=None)


def test_constraint_config_can_disable_individual_steps() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({"gen_solar": [-5.0, 10, -1.0, 5], "carbon_ets": [50.0] * 4}, index=idx)
    raw = df.copy()
    # Disable everything
    out = apply_all(
        df,
        raw_df=raw,
        config=ConstraintConfig(
            clip_generation=False,
            enforce_solar_night=False,
            enforce_flow_coherence=False,
            restore_carbon_ets=False,
            add_structural_zero=False,
        ),
    )
    pd.testing.assert_frame_equal(out, df)
