import numpy as np
import pandas as pd
import pytest

from preprocessing.impute import (
    NaNScaler, apply_domain_fixes, make_windows,
    aggregate_predictions, recompute_net_imports,
)


def test_nan_scaler_fit_ignores_nan():
    arr = np.array([[1.0, np.nan], [3.0, 2.0], [np.nan, 4.0]])
    scaler = NaNScaler().fit(arr)
    assert scaler.mean_[0] == pytest.approx(2.0)   # mean([1, 3])
    assert scaler.mean_[1] == pytest.approx(3.0)   # mean([2, 4])


def test_nan_scaler_transform_preserves_nan():
    arr = np.array([[1.0, np.nan], [3.0, 2.0]])
    scaler = NaNScaler().fit(arr)
    transformed = scaler.transform(arr)
    assert np.isnan(transformed[0, 1])


def test_nan_scaler_roundtrip():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    scaler = NaNScaler().fit(arr)
    roundtripped = scaler.inverse_transform(scaler.transform(arr))
    np.testing.assert_allclose(roundtripped, arr)


def test_nan_scaler_zero_variance_column():
    # column of all 5s - scale_ must not be 0 (would cause divide-by-zero)
    arr = np.array([[5.0], [5.0], [5.0]])
    scaler = NaNScaler().fit(arr)
    assert scaler.scale_[0] == pytest.approx(1.0)


def test_nan_scaler_all_nan_column():
    # column 1 is entirely NaN - scale_ must be 1.0, not NaN
    arr = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, np.nan]])
    scaler = NaNScaler().fit(arr)
    assert not np.isnan(scaler.scale_[1])
    assert scaler.scale_[1] == pytest.approx(1.0)


def test_domain_fix_coal_gas_zero_filled():
    idx = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"gen_fossil_coal_gas": [np.nan, 1.0, np.nan],
         "gen_nuclear": [100.0, 100.0, 100.0],
         "carbon_ets": [np.nan, np.nan, np.nan]},
        index=idx,
    )
    fixed = apply_domain_fixes(df)
    assert fixed["gen_fossil_coal_gas"].isna().sum() == 0
    assert fixed["gen_fossil_coal_gas"].iloc[0] == pytest.approx(0.0)
    assert fixed["gen_fossil_coal_gas"].iloc[1] == pytest.approx(1.0)  # observed kept


def test_domain_fix_nuclear_post_shutdown_only():
    idx = pd.date_range("2023-04-15", periods=3, freq="D", tz="UTC")
    # idx[0] = 2023-04-15 (on shutdown day - NOT strictly after → keep)
    # idx[1] = 2023-04-16 (after → NaN → zero)
    # idx[2] = 2023-04-17 (after → NaN → zero)
    df = pd.DataFrame(
        {"gen_fossil_coal_gas": [0.0, 0.0, 0.0],
         "gen_nuclear": [100.0, np.nan, np.nan],
         "carbon_ets": [1.0, 1.0, 1.0]},
        index=idx,
    )
    fixed = apply_domain_fixes(df)
    assert fixed["gen_nuclear"].iloc[0] == pytest.approx(100.0)  # on day - kept
    assert fixed["gen_nuclear"].iloc[1] == pytest.approx(0.0)    # after - zero-filled
    assert fixed["gen_nuclear"].iloc[2] == pytest.approx(0.0)    # after - zero-filled


def test_domain_fix_carbon_ets_untouched():
    idx = pd.date_range("2019-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"gen_fossil_coal_gas": [np.nan, np.nan, np.nan],
         "gen_nuclear": [100.0, 100.0, 100.0],
         "carbon_ets": [np.nan, np.nan, np.nan]},
        index=idx,
    )
    fixed = apply_domain_fixes(df)
    assert fixed["carbon_ets"].isna().all()


def test_domain_fix_does_not_modify_original():
    idx = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"gen_fossil_coal_gas": [np.nan, np.nan],
         "gen_nuclear": [np.nan, np.nan],
         "carbon_ets": [np.nan, np.nan]},
        index=idx,
    )
    _ = apply_domain_fixes(df)
    assert df["gen_fossil_coal_gas"].isna().all()  # original unchanged


def test_make_windows_shape_non_overlapping():
    arr = np.zeros((200, 5), dtype=np.float64)
    windows, starts = make_windows(arr, T=10, stride=10)
    assert windows.shape == (20, 10, 5)
    assert len(starts) == 20


def test_make_windows_shape_overlapping():
    arr = np.zeros((25, 3), dtype=np.float64)
    windows, starts = make_windows(arr, T=10, stride=5)
    # starts: 0, 5, 10, 15  (last start ≤ 25-10=15)
    assert windows.shape == (4, 10, 3)
    assert starts[0] == 0
    assert starts[3] == 15


def test_make_windows_content():
    arr = np.arange(30, dtype=np.float64).reshape(30, 1)
    windows, starts = make_windows(arr, T=10, stride=10)
    np.testing.assert_array_equal(windows[0, :, 0], np.arange(10))
    np.testing.assert_array_equal(windows[1, :, 0], np.arange(10, 20))


def test_make_windows_preserves_nan():
    arr = np.full((20, 3), np.nan)
    arr[10:20, :] = 1.0
    windows, _ = make_windows(arr, T=10, stride=10)
    assert np.isnan(windows[0]).all()        # window 0-9: all NaN
    assert not np.isnan(windows[1]).any()    # window 10-19: all 1.0


def test_make_windows_dtype_float32():
    arr = np.ones((20, 2), dtype=np.float64)
    windows, _ = make_windows(arr, T=10, stride=10)
    assert windows.dtype == np.float32


def test_aggregate_non_overlapping():
    T, N = 4, 2
    predictions = np.array(
        [[[1, 2], [3, 4], [5, 6], [7, 8]],
         [[9, 10], [11, 12], [13, 14], [15, 16]]],
        dtype=np.float64,
    )
    starts = np.array([0, 4])
    result = aggregate_predictions(predictions, starts, total_len=8, N=N, T=T)
    assert result.shape == (8, 2)
    np.testing.assert_allclose(result[0], [1.0, 2.0])
    np.testing.assert_allclose(result[7], [15.0, 16.0])


def test_aggregate_overlapping_averages():
    T, N = 4, 1
    # window 0 covers positions 0-3, all value 10
    # window 1 covers positions 2-5, all value 20
    # positions 2-3 covered by both → average = 15
    preds = np.array(
        [[[10.0], [10.0], [10.0], [10.0]],
         [[20.0], [20.0], [20.0], [20.0]]],
        dtype=np.float64,
    )
    starts = np.array([0, 2])
    result = aggregate_predictions(preds, starts, total_len=6, N=N, T=T)
    assert result[0, 0] == pytest.approx(10.0)
    assert result[1, 0] == pytest.approx(10.0)
    assert result[2, 0] == pytest.approx(15.0)
    assert result[3, 0] == pytest.approx(15.0)
    assert result[4, 0] == pytest.approx(20.0)
    assert result[5, 0] == pytest.approx(20.0)


def test_aggregate_output_shape():
    T, N = 5, 3
    preds = np.zeros((10, T, N))
    starts = np.arange(0, 10 * T, T)
    result = aggregate_predictions(preds, starts, total_len=50, N=N, T=T)
    assert result.shape == (50, 3)


def test_recompute_net_imports_values():
    idx = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"FR_to_DELU": [100.0, 200.0, 50.0],
         "DELU_to_FR": [30.0, 80.0, 70.0],
         "AT_to_DELU": [10.0, 20.0, 30.0],
         "DELU_to_AT": [5.0, 10.0, 15.0],
         "CH_to_DELU": [40.0, 60.0, 80.0],
         "DELU_to_CH": [20.0, 30.0, 40.0]},
        index=idx,
    )
    result = recompute_net_imports(df)
    np.testing.assert_allclose(result["net_import_FR"].values, [70.0, 120.0, -20.0])
    np.testing.assert_allclose(result["net_import_AT"].values, [5.0, 10.0, 15.0])
    np.testing.assert_allclose(result["net_import_CH"].values, [20.0, 30.0, 40.0])


def test_recompute_net_imports_does_not_modify_original():
    idx = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"FR_to_DELU": [100.0, 200.0], "DELU_to_FR": [30.0, 80.0],
         "AT_to_DELU": [10.0, 20.0], "DELU_to_AT": [5.0, 10.0],
         "CH_to_DELU": [40.0, 60.0], "DELU_to_CH": [20.0, 30.0]},
        index=idx,
    )
    _ = recompute_net_imports(df)
    assert "net_import_FR" not in df.columns


def test_recompute_output_has_50_cols_from_47():
    idx = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    base_cols = {c: [0.0, 0.0] for c in
                 ["FR_to_DELU", "DELU_to_FR", "AT_to_DELU",
                  "DELU_to_AT", "CH_to_DELU", "DELU_to_CH"]}
    # pad to 47 columns
    for i in range(41):
        base_cols[f"col_{i}"] = [0.0, 0.0]
    df = pd.DataFrame(base_cols, index=idx)
    result = recompute_net_imports(df)
    assert result.shape[1] == 50
