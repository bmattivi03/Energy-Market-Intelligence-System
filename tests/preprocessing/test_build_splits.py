import pandas as pd
import numpy as np

from preprocessing.build_splits import apply_post_imputation_fixes, split_frame


def _toy():
    idx = pd.date_range("2019-01-01", "2025-03-31 23:00", freq="h", tz="UTC")
    df = pd.DataFrame(
        {"carbon_ets": np.nan, "gen_fossil_coal_gas": 5.0, "price": 1.0},
        index=idx,
    )
    df.loc[df.index[100:], "carbon_ets"] = 2.0  # leave first 100 NaN for bfill
    mask = pd.DataFrame(False, index=idx, columns=df.columns)
    mask.loc[mask.index < "2023-01-01", "gen_fossil_coal_gas"] = True  # structural
    return df, mask


def test_carbon_ets_no_nan_after_fix():
    df, mask = _toy()
    out = apply_post_imputation_fixes(df, mask)
    assert out["carbon_ets"].isna().sum() == 0


def test_structural_zero_indicator_and_zeroing():
    df, mask = _toy()
    out = apply_post_imputation_fixes(df, mask)
    assert "gen_fossil_coal_gas_structural_zero" in out.columns
    struct = out["gen_fossil_coal_gas_structural_zero"] == 1.0
    assert (out.loc[struct, "gen_fossil_coal_gas"] == 0.0).all()
    assert out["gen_fossil_coal_gas_structural_zero"].dtype == float


def test_split_bounds_exclusive_inclusive():
    df, mask = _toy()
    out = apply_post_imputation_fixes(df, mask)
    train, val, test = split_frame(out)
    assert train.index.max() < pd.Timestamp("2024-01-01", tz="UTC")
    assert val.index.min() == pd.Timestamp("2024-01-01", tz="UTC")
    assert val.index.max() < pd.Timestamp("2025-01-01", tz="UTC")
    assert test.index.min() == pd.Timestamp("2025-01-01", tz="UTC")
    assert test.index.max() < pd.Timestamp("2025-04-01", tz="UTC")
    assert len(test) == 2160  # Jan-Mar 2025 hourly
