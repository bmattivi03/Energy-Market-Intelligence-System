"""Tests for module_a.evaluation metric functions.

Hand-constructed tiny inputs with values verifiable by manual calculation.
"""

import numpy as np
import pandas as pd
import pytest

from module_a import evaluation as ev


# ---------------------------------------------------------------- point metrics


def test_mae_manual():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 4.0, 0.0]  # abs errors: 0, 2, 3 -> mean 5/3
    assert ev.mae(y_true, y_pred) == pytest.approx(5.0 / 3.0)


def test_rmse_manual():
    y_true = [0.0, 0.0, 0.0]
    y_pred = [3.0, 4.0, 0.0]  # sq errors 9,16,0 -> mean 25/3 -> sqrt
    assert ev.rmse(y_true, y_pred) == pytest.approx(np.sqrt(25.0 / 3.0))


def test_mae_zero_when_perfect():
    y = np.array([5.0, -2.0, 7.5])
    assert ev.mae(y, y) == 0.0
    assert ev.rmse(y, y) == 0.0


def test_nmae_manual():
    y_true = [10.0, 20.0, 30.0]      # mean abs = 20
    y_pred = [12.0, 18.0, 33.0]      # abs err 2,2,3 -> mean 7/3
    assert ev.nmae(y_true, y_pred) == pytest.approx((7.0 / 3.0) / 20.0)


# ---------------------------------------------------------------- pinball


def test_pinball_under_prediction():
    """Under-predicting (err>0) is weighted by q."""
    y_true = [10.0]
    y_pred = [8.0]  # err = +2
    q = 0.9
    # loss = q * err = 0.9 * 2 = 1.8
    assert ev.pinball(y_true, y_pred, q) == pytest.approx(1.8)


def test_pinball_over_prediction():
    """Over-predicting (err<0) is weighted by (q-1)."""
    y_true = [10.0]
    y_pred = [13.0]  # err = -3
    q = 0.9
    # loss = (q-1)*err = (-0.1)*(-3) = 0.3
    assert ev.pinball(y_true, y_pred, q) == pytest.approx(0.3)


def test_pinball_median_is_half_mae():
    """At q=0.5 the pinball loss equals 0.5 * MAE."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = rng.normal(size=50)
    assert ev.pinball(y_true, y_pred, 0.5) == pytest.approx(
        0.5 * ev.mae(y_true, y_pred)
    )


def test_pinball_non_negative():
    rng = np.random.default_rng(1)
    for q in (0.1, 0.5, 0.9):
        y_true = rng.normal(size=100)
        y_pred = rng.normal(size=100)
        assert ev.pinball(y_true, y_pred, q) >= 0.0


# ---------------------------------------------------------------- coverage / width


def test_coverage_manual():
    y_true = [1.0, 5.0, 9.0, 11.0]
    q_low = [0.0, 0.0, 0.0, 0.0]
    q_high = [10.0, 10.0, 10.0, 10.0]
    # inside [0,10]: 1,5,9 -> yes (3); 11 -> no. coverage = 3/4
    assert ev.coverage(y_true, q_low, q_high) == pytest.approx(0.75)


def test_coverage_in_unit_interval():
    rng = np.random.default_rng(2)
    y = rng.normal(size=200)
    lo = y - rng.random(200)
    hi = y + rng.random(200)
    c = ev.coverage(y, lo, hi)
    assert 0.0 <= c <= 1.0


def test_coverage_endpoints_inclusive():
    # Values exactly on the bounds count as covered.
    assert ev.coverage([0.0, 10.0], [0.0, 0.0], [10.0, 10.0]) == 1.0


def test_mean_interval_width_manual():
    q_low = [1.0, 2.0]
    q_high = [4.0, 10.0]  # widths 3, 8 -> mean 5.5
    assert ev.mean_interval_width(q_low, q_high) == pytest.approx(5.5)


# ---------------------------------------------------------------- winkler


def test_winkler_inside_equals_width():
    """When the actual lies inside the interval, score == interval width."""
    y_true = [5.0]
    lo, hi = [0.0], [10.0]
    assert ev.winkler_score(y_true, lo, hi, alpha=0.2) == pytest.approx(10.0)


def test_winkler_penalises_misses():
    """A miss adds a positive penalty on top of the width."""
    inside = ev.winkler_score([5.0], [0.0], [10.0], alpha=0.2)
    miss_above = ev.winkler_score([20.0], [0.0], [10.0], alpha=0.2)
    # width 10 + (2/0.2)*(20-10) = 10 + 100 = 110
    assert miss_above == pytest.approx(110.0)
    assert miss_above > inside


# ---------------------------------------------------------------- empirical coverage


def test_empirical_coverage_at_quantiles():
    # 10 actuals 1..10; predicted quantiles flat.
    y_true = list(range(1, 11))
    q10 = [3.0] * 10   # actuals < 3 : {1,2} -> 0.2
    q50 = [5.5] * 10   # actuals < 5.5: {1..5} -> 0.5
    q90 = [9.0] * 10   # actuals < 9 : {1..8} -> 0.8
    res = ev.empirical_coverage_at_quantiles(y_true, q10, q50, q90)
    assert res["below_q10"] == pytest.approx(0.2)
    assert res["below_q50"] == pytest.approx(0.5)
    assert res["below_q90"] == pytest.approx(0.8)


# ---------------------------------------------------------------- Diebold-Mariano


def test_diebold_mariano_identical_errors():
    """Identical error series -> zero variance -> DM=0, p=1 (the guard branch)."""
    e = np.array([1.0, 2.0, 3.0, 4.0])
    stat, p = ev.diebold_mariano(e, e)
    assert stat == 0.0
    assert p == 1.0


def test_diebold_mariano_sign_and_pvalue():
    """Model errors systematically smaller -> negative DM stat, p in [0,1]."""
    rng = np.random.default_rng(7)
    errs_model = np.abs(rng.normal(0, 1, 300))
    errs_base = errs_model + 0.5  # baseline strictly worse
    stat, p = ev.diebold_mariano(errs_model, errs_base)
    assert stat < 0.0          # model better -> negative
    assert 0.0 <= p <= 1.0
    assert p < 0.05            # large, consistent gap is significant


# ---------------------------------------------------------------- baselines


def test_naive_forecast_uses_lag24():
    idx = pd.date_range("2023-01-01", periods=72, freq="h", tz="UTC")
    load = pd.Series(np.arange(72, dtype=float), index=idx)
    preds = ev.naive_forecast(load, horizon=ev.HORIZON)
    # For origin t and horizon h, prediction = load at (t + h - 24).
    origin = idx[40]
    h = 5
    expected_anchor = origin + pd.Timedelta(hours=h - 24)
    assert preds.loc[(origin, h), "pred"] == pytest.approx(load[expected_anchor])


def test_seasonal_naive_forecast_uses_lag168():
    idx = pd.date_range("2023-01-01", periods=400, freq="h", tz="UTC")
    load = pd.Series(np.arange(400, dtype=float), index=idx)
    preds = ev.seasonal_naive_forecast(load, horizon=ev.HORIZON)
    origin = idx[200]
    h = 3
    expected_anchor = origin + pd.Timedelta(hours=h - 168)
    assert preds.loc[(origin, h), "pred"] == pytest.approx(load[expected_anchor])
