import numpy as np
import pandas as pd

from module_joint.evaluate import compare_price, dm_power, make_truth


def _pred(origins, horizons, center):
    idx = pd.MultiIndex.from_product(
        [origins, horizons], names=["origin_ts", "horizon_h"]
    )
    n = len(idx)
    return pd.DataFrame(
        {"q10": center - 5, "q50": np.full(n, center), "q90": center + 5}, index=idx
    )


def test_make_truth_aligns_to_target_time():
    idx = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    df = pd.DataFrame({"price": np.arange(30.0)}, index=idx)
    pidx = pd.MultiIndex.from_tuples(
        [(idx[0], 1), (idx[0], 2)], names=["origin_ts", "horizon_h"]
    )
    truth = make_truth(df, pidx, "price")
    assert truth.iloc[0] == 1.0 and truth.iloc[1] == 2.0


def test_compare_price_metrics_and_segments():
    origins = pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC")
    horizons = list(range(1, 25))
    pred = _pred(origins, horizons, center=50.0)
    truth = pd.Series(50.0, index=pred.index)
    res = compare_price(pred, truth)
    assert res["overall"]["mae"] == 0.0
    assert res["overall"]["coverage"] == 1.0
    assert set(res["segments"]) == {"h1_6", "h7_18", "h19_24"}


def test_compare_price_dm_vs_baseline():
    origins = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    horizons = [1]
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1.0, len(origins))
    truth_vals = 50.0 + noise
    pred = _pred(origins, horizons, center=50.0)  # close to truth
    base = _pred(origins, horizons, center=54.0)  # biased high -> larger errors
    truth = pd.Series(truth_vals, index=pred.index)
    res = compare_price(pred, truth, baseline_pred=base)
    assert "dm_vs_baseline" in res
    assert res["dm_vs_baseline"]["statistic"] < 0  # model loss < baseline loss
    assert res["dm_vs_baseline"]["p_value"] < 0.05


def test_dm_power_scales_with_n():
    p_small = dm_power(100, sd_loss_diff=10.0)
    p_large = dm_power(2160, sd_loss_diff=10.0)
    assert p_small > p_large > 0
