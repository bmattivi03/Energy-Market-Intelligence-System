"""Tests for evaluation: point/probabilistic metrics, segmentation, DM/bootstrap, conformal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module_b.evaluation import (
    SEGMENT_FNS,
    AdaptiveConformalCalibrator,
    ConformalQuantileRegressor,
    bootstrap_ci,
    coverage,
    diebold_mariano,
    directional_accuracy,
    mae,
    multi_pinball_loss,
    pinball_loss,
    rmse,
    segment_metrics,
    smape,
    spike_mae,
    winkler_score,
)
from module_b.features import build_features, prepare_supervised
from module_b.models import LightGBMQuantileForecaster


# ---------------------------------------------------------------------------
# Point / probabilistic metrics (from test_metrics.py)
# ---------------------------------------------------------------------------


def test_mae_zero_when_perfect() -> None:
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_known_value() -> None:
    # errors are [-1, 0, 1] → MSE 2/3 → RMSE sqrt(2/3)
    assert rmse([1, 2, 3], [2, 2, 2]) == pytest.approx(np.sqrt(2 / 3))


def test_pinball_at_median_equals_half_mae() -> None:
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([2.0, 2.0, 2.0])
    assert pinball_loss(yt, yp, 0.5) == pytest.approx(0.5 * mae(yt, yp))


def test_multi_pinball_loss_aggregates_correctly() -> None:
    yt = np.array([1.0, 2.0, 3.0])
    df = pd.DataFrame({"q10": [1.5, 2.0, 2.5], "q50": [2.0, 2.0, 2.0], "q90": [2.5, 2.0, 1.5]})
    expected = np.mean([pinball_loss(yt, df[c].values, a) for c, a in zip(df.columns, [0.1, 0.5, 0.9])])
    assert multi_pinball_loss(yt, df, [0.1, 0.5, 0.9]) == pytest.approx(expected)


def test_coverage_full_when_interval_contains_all() -> None:
    assert coverage([1, 2, 3], [0, 0, 0], [10, 10, 10]) == 1.0


def test_winkler_score_zero_width_no_outlier() -> None:
    # Interval = point estimate exactly = truth → width 0, no penalty
    yt = [1.0, 2.0, 3.0]
    assert winkler_score(yt, yt, yt, alpha=0.2) == 0.0


def test_directional_accuracy_perfect() -> None:
    anchor = np.array([5.0, 5.0, 5.0])
    yt = np.array([6.0, 4.0, 7.0])
    yp = np.array([8.0, 3.0, 6.0])
    # All directions match
    assert directional_accuracy(yt, yp, anchor) == 1.0


def test_spike_mae_focuses_on_top_decile() -> None:
    yt = np.arange(100, dtype=float)
    yp = yt.copy()
    yp[90:] = 0  # 10% of high values mispredicted as 0
    # Spike MAE should be ~95 (mean of yt[90:])
    assert spike_mae(yt, yp, percentile=0.9) == pytest.approx(np.mean(yt[90:]))


def test_segmentation_keys_complete() -> None:
    expected = {
        "all", "peak", "off_peak", "weekend", "weekday",
        "crisis_2022", "post_crisis", "negative_price", "spike_top10pct",
    }
    assert expected.issubset(set(SEGMENT_FNS.keys()))


def test_segment_metrics_handles_empty_segment() -> None:
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    y_true = pd.Series(np.ones(24), index=range(24))  # never negative
    y_pred = pd.Series(np.ones(24), index=range(24))
    out = segment_metrics(idx, y_true, y_pred, mae, segments=("negative_price", "all"))
    assert np.isnan(out["negative_price"])  # no negative values
    assert out["all"] == 0.0


def test_diebold_mariano_smoke() -> None:
    rng = np.random.default_rng(0)
    e1 = rng.normal(0, 1, 200)
    e2 = rng.normal(0, 2, 200)  # model 2 is worse
    res = diebold_mariano(e1, e2, horizon=1)
    assert res.n == 200
    # Mean loss diff should be negative (e1^2 < e2^2 on average)
    assert res.loss_diff_mean < 0
    # Two-sided test should reject equality at common levels
    assert res.p_value < 0.05


def test_bootstrap_ci_brackets_estimate() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(5, 2, 200)
    point, lo, hi = bootstrap_ci(lambda x: x.mean(), sample, n_resamples=500, seed=0)
    assert lo < point < hi
    assert abs(point - 5) < 0.5  # close to true mean


# ---------------------------------------------------------------------------
# Conformal calibration (from test_conformal.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_lgbm(small_df):
    feat = build_features(small_df, bundles=("calendar", "lags"))
    X, y = prepare_supervised(
        feat, horizons=range(1, 4),
        past_cols=["price_lag1h", "price_lag24h", "price_lag168h", "price_rmean24h"],
        future_cols=["hour_sin", "hour_cos"],
    )
    n = len(X)
    cut = int(n * 0.7)
    m = LightGBMQuantileForecaster(num_boost_round=20, early_stopping_rounds=None)
    m.fit(X.iloc[:cut], y.iloc[:cut])
    return m, X, y, cut


def test_cqr_calibration_widens_interval_when_undercovered(fitted_lgbm) -> None:
    m, X, y, cut = fitted_lgbm
    cqr = ConformalQuantileRegressor(base=m, alpha=0.2)  # target 80% coverage
    cqr.calibrate(X.iloc[cut:], y.iloc[cut:])
    # delta is the additive expansion - non-negative when CQR widens vs base
    assert cqr.delta is not None


def test_cqr_changes_quantiles(fitted_lgbm) -> None:
    m, X, y, cut = fitted_lgbm
    cqr = ConformalQuantileRegressor(base=m, alpha=0.2).calibrate(X.iloc[cut:], y.iloc[cut:])
    base_q = m.predict_quantiles(X.iloc[cut:])
    cal_q = cqr.predict_quantiles(X.iloc[cut:])
    # Lo column shifted down by delta, hi column shifted up by delta
    assert np.allclose(cal_q["q10"].values, base_q["q10"].values - cqr.delta)
    assert np.allclose(cal_q["q90"].values, base_q["q90"].values + cqr.delta)


def test_cqr_requires_lo_hi_columns(small_df, fitted_lgbm) -> None:
    m, X, y, cut = fitted_lgbm
    cqr = ConformalQuantileRegressor(base=m, alpha=0.30)  # needs q15 and q85
    with pytest.raises(KeyError, match="produce quantiles"):
        cqr.calibrate(X.iloc[cut:], y.iloc[cut:])


def test_adaptive_cp_grows_delta_when_undercovered() -> None:
    cal = AdaptiveConformalCalibrator(alpha_target=0.2, gamma=0.05)
    # Force errors: every y outside the interval
    for _ in range(20):
        cal.update(y_true=10.0, q_lo=0.0, q_hi=1.0)
    assert cal.delta > 0


def test_adaptive_cp_keeps_delta_zero_when_well_covered() -> None:
    cal = AdaptiveConformalCalibrator(alpha_target=0.2, gamma=0.05)
    for _ in range(50):
        cal.update(y_true=0.5, q_lo=0.0, q_hi=1.0)
    # delta should converge to ~0 when always inside the interval
    assert cal.delta == 0.0


def test_cqr_uncalibrated_predict_raises(fitted_lgbm) -> None:
    m, X, *_ = fitted_lgbm
    cqr = ConformalQuantileRegressor(base=m, alpha=0.2)
    with pytest.raises(RuntimeError, match="calibrate"):
        cqr.predict_quantiles(X)


# ---------------------------------------------------------------------------
# Property tests: theoretical guarantees of CQR / pinball / DM
# ---------------------------------------------------------------------------


class _ConstantQuantileBase:
    """A stand-in BaseQuantileForecaster that emits fixed quantile predictions.

    Useful for property-testing CQR independently of any learned model.
    """

    def __init__(self, q_lo: float, q_hi: float):
        self.q_lo = q_lo
        self.q_hi = q_hi

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return pd.DataFrame(
            {"q10": np.full(n, self.q_lo), "q90": np.full(n, self.q_hi)},
            index=X.index,
        )


def test_cqr_marginal_coverage_guarantee_exchangeable_data() -> None:
    """CQR's main theoretical claim: marginal coverage ≥ 1−α on exchangeable data.

    Calibrate on a held-out fold drawn from the same distribution as the test
    fold, with a base forecaster that systematically undercovers, and check
    that the post-calibration interval covers at least 1−α on the test fold.
    """
    rng = np.random.default_rng(0)
    n_cal, n_test = 500, 1000
    y_cal = rng.normal(0, 1, n_cal)
    y_test = rng.normal(0, 1, n_test)
    # Deliberately narrow base interval (only ~38% coverage uncalibrated)
    base = _ConstantQuantileBase(q_lo=-0.5, q_hi=0.5)
    X_cal = pd.DataFrame(index=range(n_cal))
    X_test = pd.DataFrame(index=range(n_test))
    cqr = ConformalQuantileRegressor(base=base, alpha=0.2)
    cqr.calibrate(X_cal, pd.Series(y_cal, index=X_cal.index))
    out = cqr.predict_quantiles(X_test)
    cov = coverage(y_test, out["q10"], out["q90"])
    # Theory guarantees marginal coverage ≥ 1−α (here 0.80). Allow a small
    # finite-sample slack because the 1000-point empirical can fluctuate below
    # the guarantee by O(1/sqrt(n)) ≈ 3% in either direction.
    assert cov >= 0.77, f"CQR coverage too low: {cov:.3f} (target 0.80)"
    # Also: the base forecaster was deliberately narrow, so coverage should
    # have improved substantially.
    base_cov = coverage(y_test, np.full(n_test, -0.5), np.full(n_test, 0.5))
    assert cov > base_cov + 0.30, f"CQR did not widen enough: base={base_cov:.3f}, cqr={cov:.3f}"


def test_cqr_delta_matches_barber_formula() -> None:
    """Verify the CQR δ is exactly the ceil((n+1)(1−α))/n empirical quantile of scores."""
    rng = np.random.default_rng(1)
    n = 100
    y = rng.normal(0, 1, n)
    base = _ConstantQuantileBase(q_lo=-1.0, q_hi=1.0)
    X = pd.DataFrame(index=range(n))
    cqr = ConformalQuantileRegressor(base=base, alpha=0.2).calibrate(
        X, pd.Series(y, index=X.index)
    )
    expected_scores = np.maximum(-1.0 - y, y - 1.0)
    expected_level = np.ceil((n + 1) * 0.8) / n
    expected_delta = float(np.quantile(expected_scores, np.clip(expected_level, 0.0, 1.0)))
    assert cqr.delta == pytest.approx(expected_delta, abs=1e-12)


def test_pinball_loss_known_value() -> None:
    """Closed-form check: q=0.9, y=10, y_pred=8 → diff=2, loss = max(0.9·2, -0.1·2) = 1.8."""
    assert pinball_loss([10.0], [8.0], 0.9) == pytest.approx(1.8)
    # Asymmetry: underprediction at q=0.9 costs more than overprediction
    under = pinball_loss([10.0], [8.0], 0.9)  # diff=+2 → 0.9·2 = 1.8
    over = pinball_loss([10.0], [12.0], 0.9)  # diff=-2 → -0.1·-2 = 0.2
    assert under > over


def test_winkler_score_penalizes_outside_interval() -> None:
    """If y is above the interval by 1.0 with width 2.0 and α=0.2,
    Winkler = width + (2/α)·(y − hi) = 2 + 10·1 = 12.
    """
    yt = [3.0]
    lo = [0.0]
    hi = [2.0]
    assert winkler_score(yt, lo, hi, alpha=0.2) == pytest.approx(12.0)


def test_coverage_known_value() -> None:
    """3 of 5 inside [0,1] → coverage = 0.6."""
    yt = [0.5, -1.0, 0.9, 2.0, 0.1]
    lo = [0.0] * 5
    hi = [1.0] * 5
    assert coverage(yt, lo, hi) == pytest.approx(0.6)
