"""Unit tests for src/preprocessing/imputation_eval.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.imputation_eval import (
    GENERATION_NONNEG_COLS,
    MaskedColumnReport,
    StaticColumnReport,
    artificial_mask_audit,
    compute_sun_elevation,
    count_physical_violations,
    static_audit,
    summarize_masked_to_markdown,
    summarize_static_to_markdown,
)


@pytest.fixture
def synthetic_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=24 * 14, freq="h", tz="UTC")


@pytest.fixture
def synthetic_raw(synthetic_index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(synthetic_index)
    df = pd.DataFrame(
        {
            "gen_solar": np.maximum(0, np.sin(np.arange(n) / 24 * 2 * np.pi) * 5000)
            + rng.normal(0, 50, n),
            "gen_wind_onshore": rng.uniform(1000, 8000, n),
            "FR_to_DELU": rng.uniform(0, 3000, n),
            "DELU_to_FR": rng.uniform(0, 3000, n),
            "carbon_ets": rng.uniform(50, 100, n),
        },
        index=synthetic_index,
    )
    # Inject some NaNs (MCAR-style)
    for col in df.columns:
        idx = rng.choice(n, size=n // 5, replace=False)
        df.loc[df.index[idx], col] = np.nan
    return df


def test_compute_sun_elevation_is_negative_at_night(synthetic_index: pd.DatetimeIndex) -> None:
    elev = compute_sun_elevation(synthetic_index)
    midnight = synthetic_index.hour == 0
    noon = synthetic_index.hour == 12
    assert (elev[midnight] < 0).all(), "sun must be below horizon at midnight UTC for Germany"
    assert (elev[noon] > 0).all(), "sun must be above horizon at 12 UTC in mid-Jan for Germany"


def test_count_physical_violations_negative_generation() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"gen_wind_onshore": [-100.0, 0, 50, -1.0, 200, 0, 0, 0, 0, 0]}, index=idx)
    counts = count_physical_violations(df, "gen_wind_onshore")
    assert counts["negative_generation"] == 2  # -100 and -1.0


def test_count_physical_violations_solar_at_night(synthetic_index: pd.DatetimeIndex) -> None:
    df = pd.DataFrame({"gen_solar": np.full(len(synthetic_index), 1000.0)}, index=synthetic_index)
    elev = compute_sun_elevation(synthetic_index)
    counts = count_physical_violations(df, "gen_solar", elev)
    assert counts["solar_at_full_night"] > 0


def test_count_physical_violations_flow_both_positive() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"FR_to_DELU": [1000.0, 0, 500, 0, 100], "DELU_to_FR": [500.0, 0, 0, 100, 200]},
        index=idx,
    )
    counts = count_physical_violations(df, "FR_to_DELU")
    assert counts["both_directions_positive"] == 2  # rows 0 and 4


def test_static_audit_skips_columns_without_nans(synthetic_raw: pd.DataFrame) -> None:
    no_nan_raw = synthetic_raw.fillna(0)
    imputed = no_nan_raw.copy()
    reports = static_audit(imputed, no_nan_raw)
    assert reports == []


def test_static_audit_flags_sign_inversion(synthetic_raw: pd.DataFrame) -> None:
    imputed = synthetic_raw.copy()
    # Imputed cells are sign-inverted vs observed cells
    nan_mask = synthetic_raw["FR_to_DELU"].isna()
    imputed.loc[nan_mask, "FR_to_DELU"] = -2000.0  # observed mean ~+1500
    imputed = imputed.ffill().bfill()
    reports = {r.column: r for r in static_audit(imputed, synthetic_raw)}
    fr = reports["FR_to_DELU"]
    assert fr.status == "red"
    assert any("sign" in r or "shift" in r for r in fr.reasons)


def test_static_audit_flags_variance_collapse(synthetic_raw: pd.DataFrame) -> None:
    imputed = synthetic_raw.copy()
    nan_mask = synthetic_raw["gen_wind_onshore"].isna()
    imputed.loc[nan_mask, "gen_wind_onshore"] = 4500.0  # constant → collapsed std
    imputed = imputed.ffill().bfill()
    reports = {r.column: r for r in static_audit(imputed, synthetic_raw)}
    wind = reports["gen_wind_onshore"]
    assert wind.status == "red"
    assert any("variance" in r for r in wind.reasons)


def test_artificial_mask_audit_perfect_imputer(synthetic_raw: pd.DataFrame) -> None:
    """A perfect oracle (returns the same df) should score MAE = 0."""

    def oracle(df: pd.DataFrame) -> pd.DataFrame:
        # The "ground truth" is synthetic_raw; the masked df has *extra* NaNs.
        # An oracle imputer would fill those extras with the original values.
        out = df.copy()
        for col in df.columns:
            mask = df[col].isna() & synthetic_raw[col].notna()
            out.loc[mask, col] = synthetic_raw.loc[mask, col]
        return out

    reports, mask_record = artificial_mask_audit(
        raw_df=synthetic_raw,
        imputer=oracle,
        eval_index=synthetic_raw.index,
        mask_ratio=0.2,
        seed=42,
    )
    assert len(reports) > 0
    for r in reports:
        assert r.mae == pytest.approx(0.0, abs=1e-6)


def test_artificial_mask_audit_constant_imputer_has_nonzero_error(synthetic_raw: pd.DataFrame) -> None:
    def constant(df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(0.0)

    reports, _ = artificial_mask_audit(
        raw_df=synthetic_raw,
        imputer=constant,
        eval_index=synthetic_raw.index,
        mask_ratio=0.2,
        seed=42,
    )
    assert len(reports) > 0
    assert all(r.mae > 0 for r in reports)


def test_summarize_static_to_markdown_renders(synthetic_raw: pd.DataFrame) -> None:
    imputed = synthetic_raw.copy().ffill().bfill()
    reports = static_audit(imputed, synthetic_raw)
    md = summarize_static_to_markdown(reports, title="Test")
    assert "# Test" in md
    assert "| Status |" in md
    assert "Summary:" in md


def test_summarize_masked_to_markdown_renders() -> None:
    reports = [
        MaskedColumnReport(column="x", n_evaluated=100, mae=1.5, rmse=2.0, smape=0.05, pearson=0.95),
    ]
    md = summarize_masked_to_markdown(reports)
    assert "| `x` |" in md
    assert "1.5" in md


def test_classification_constants_are_consistent() -> None:
    # gen_hydro_pumped is signed and must NOT be flagged for negativity
    assert "gen_hydro_pumped" not in GENERATION_NONNEG_COLS
    # gen_solar must be in the non-negative list
    assert "gen_solar" in GENERATION_NONNEG_COLS
