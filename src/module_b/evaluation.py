"""Forecast metrics, segmentation, statistical tests, and conformal calibration.

Merges what used to live in :mod:`module_b.metrics` and :mod:`module_b.calibration`.

* Point metrics: :func:`mae`, :func:`rmse`, :func:`smape`,
  :func:`directional_accuracy`, :func:`spike_mae`.
* Probabilistic: :func:`pinball_loss`, :func:`multi_pinball_loss`,
  :func:`coverage`, :func:`winkler_score`.
* Segmentation: :data:`SEGMENT_FNS`, :func:`segment_metrics`.
* Significance: :func:`diebold_mariano`, :func:`bootstrap_ci`.
* Calibration: :class:`ConformalQuantileRegressor`, :class:`AdaptiveConformalCalibrator`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from module_b.models import BaseQuantileForecaster
from module_b.features import CRISIS_END, CRISIS_START


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------- point metrics

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_to_array(y_true) - _to_array(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((_to_array(y_true) - _to_array(y_pred)) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-9) -> float:
    yt = _to_array(y_true); yp = _to_array(y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2
    return float(np.mean(np.abs(yt - yp) / np.where(denom < eps, eps, denom)))


def directional_accuracy(y_true, y_pred, anchor) -> float:
    """Fraction of predictions correct in *direction* relative to ``anchor``."""
    yt = _to_array(y_true); yp = _to_array(y_pred); a = _to_array(anchor)
    return float(np.mean(np.sign(yt - a) == np.sign(yp - a)))


def spike_mae(y_true, y_pred, percentile: float = 0.90) -> float:
    """MAE restricted to the top ``1 - percentile`` of true values."""
    yt = _to_array(y_true); yp = _to_array(y_pred)
    threshold = np.quantile(yt, percentile)
    mask = yt >= threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


# ---------------------------------------------------------------- probabilistic

def pinball_loss(y_true, q_pred, alpha: float) -> float:
    """Pinball loss at a single quantile level α ∈ (0, 1)."""
    yt = _to_array(y_true); qp = _to_array(q_pred)
    diff = yt - qp
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def multi_pinball_loss(y_true, q_pred_df: pd.DataFrame, quantiles: Sequence[float]) -> float:
    """Average pinball loss over a list of quantiles.

    ``q_pred_df`` must have columns named ``q{int(α*100)}`` for each α.
    """
    losses = [
        pinball_loss(y_true, q_pred_df[f"q{int(round(a * 100))}"].to_numpy(), a)
        for a in quantiles
    ]
    return float(np.mean(losses))


def coverage(y_true, q_low_pred, q_high_pred) -> float:
    """Empirical coverage of the prediction interval [q_low, q_high]."""
    yt = _to_array(y_true)
    return float(np.mean((yt >= _to_array(q_low_pred)) & (yt <= _to_array(q_high_pred))))


def winkler_score(y_true, q_low_pred, q_high_pred, alpha: float = 0.20) -> float:
    """Winkler interval score for a (1−α)·100% prediction interval. Lower is better."""
    yt = _to_array(y_true); lo = _to_array(q_low_pred); hi = _to_array(q_high_pred)
    width = hi - lo
    penalty_low = (2 / alpha) * (lo - yt) * (yt < lo)
    penalty_high = (2 / alpha) * (yt - hi) * (yt > hi)
    return float(np.mean(width + penalty_low + penalty_high))


# ---------------------------------------------------------------- segmentation

PEAK_HOURS = set(range(8, 21))


def _as_array(x) -> np.ndarray:
    return np.asarray(x)


SEGMENT_FNS: dict[str, Callable[[pd.DatetimeIndex, pd.Series], np.ndarray]] = {
    "all": lambda idx, y: np.ones(len(idx), dtype=bool),
    "peak": lambda idx, y: _as_array(pd.DatetimeIndex(idx).hour.isin(PEAK_HOURS)),
    "off_peak": lambda idx, y: ~_as_array(pd.DatetimeIndex(idx).hour.isin(PEAK_HOURS)),
    "weekend": lambda idx, y: _as_array(pd.DatetimeIndex(idx).dayofweek >= 5),
    "weekday": lambda idx, y: _as_array(pd.DatetimeIndex(idx).dayofweek < 5),
    "crisis_2022": lambda idx, y: _as_array((idx >= CRISIS_START) & (idx < CRISIS_END)),
    "post_crisis": lambda idx, y: _as_array(idx >= CRISIS_END),
    "negative_price": lambda idx, y: (np.asarray(y) < 0),
    "spike_top10pct": lambda idx, y: (np.asarray(y) >= np.quantile(y, 0.9)),
}


def segment_metrics(
    target_ts: pd.DatetimeIndex,
    y_true: pd.Series,
    y_pred: pd.Series,
    metric_fn: Callable,
    segments: tuple[str, ...] = tuple(SEGMENT_FNS.keys()),
) -> dict[str, float]:
    """Apply ``metric_fn(y_true_seg, y_pred_seg)`` over each segment."""
    out: dict[str, float] = {}
    for seg in segments:
        mask = SEGMENT_FNS[seg](target_ts, y_true)
        if mask.sum() == 0:
            out[seg] = float("nan")
            continue
        out[seg] = float(metric_fn(y_true.to_numpy()[mask], y_pred.to_numpy()[mask]))
    return out


# ---------------------------------------------------------------- significance

@dataclass
class DMResult:
    statistic: float
    p_value: float
    n: int
    h: int
    loss_diff_mean: float


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    *,
    horizon: int = 1,
    loss: Callable[[np.ndarray], np.ndarray] | None = None,
    one_sided: bool = False,
) -> DMResult:
    """Harvey-corrected Diebold-Mariano test on two forecast error series."""
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    if e1.shape != e2.shape:
        raise ValueError("error series must have the same shape")
    loss = loss or (lambda e: e ** 2)
    d = loss(e1) - loss(e2)
    n = len(d)
    mean_d = float(d.mean())
    h = horizon
    gamma_0 = np.var(d, ddof=0)
    gammas = [np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)) for k in range(1, h)]
    var_d = (gamma_0 + 2 * sum(gammas)) / n
    if var_d <= 0:
        return DMResult(statistic=float("nan"), p_value=float("nan"), n=n, h=h, loss_diff_mean=mean_d)
    dm = mean_d / np.sqrt(var_d)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    if one_sided:
        p_value = float(stats.t.cdf(dm_hln, df=n - 1))
    else:
        p_value = float(2 * stats.t.sf(abs(dm_hln), df=n - 1))
    return DMResult(statistic=float(dm_hln), p_value=p_value, n=n, h=h, loss_diff_mean=mean_d)


def bootstrap_ci(
    fn: Callable[[np.ndarray], float],
    sample: np.ndarray,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for a scalar statistic ``fn(sample)``. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(sample)
    point = fn(sample)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        estimates[i] = fn(sample[idx])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(estimates, [alpha, 1 - alpha])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------- calibration

@dataclass
class ConformalQuantileRegressor:
    """CQR wrapper around an already-fitted base quantile forecaster.

    Usage::

        cqr = ConformalQuantileRegressor(base, alpha=0.2)
        cqr.calibrate(X_cal, y_cal)
        cqr.predict_quantiles(X_test)

    The base forecaster must already produce predictions at quantiles
    ``alpha/2`` and ``1−alpha/2`` (default 0.1 and 0.9 for 80% intervals).
    """

    base: BaseQuantileForecaster
    alpha: float = 0.20
    delta: float | None = None

    @property
    def lo_col(self) -> str:
        return f"q{int(round(self.alpha / 2 * 100))}"

    @property
    def hi_col(self) -> str:
        return f"q{int(round((1 - self.alpha / 2) * 100))}"

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> "ConformalQuantileRegressor":
        q = self.base.predict_quantiles(X_cal)
        if self.lo_col not in q.columns or self.hi_col not in q.columns:
            raise KeyError(
                f"Base forecaster must produce quantiles {self.lo_col} and {self.hi_col}; "
                f"got {list(q.columns)}"
            )
        scores = np.maximum(
            q[self.lo_col].to_numpy() - y_cal.to_numpy(),
            y_cal.to_numpy() - q[self.hi_col].to_numpy(),
        )
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.delta = float(np.quantile(scores, np.clip(level, 0.0, 1.0)))
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.delta is None:
            raise RuntimeError("must call calibrate() first")
        q = self.base.predict_quantiles(X).copy()
        q[self.lo_col] = q[self.lo_col] - self.delta
        q[self.hi_col] = q[self.hi_col] + self.delta
        return q

    def save(self, path: Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        with (path / "cqr_state.pkl").open("wb") as f:
            pickle.dump({"alpha": self.alpha, "delta": self.delta}, f)


class AdaptiveConformalCalibrator:
    """Online adaptive CP (Gibbs & Candès 2021).

    Updates a single additive correction ``delta`` each step:
        delta_{t+1} = max(0, delta_t + γ · (err_t − α_target))
    where ``err_t = 1 if y_t outside the current interval, else 0``.
    """

    def __init__(self, alpha_target: float = 0.20, gamma: float = 0.005):
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.delta = 0.0
        self._history: list[dict] = []

    def update(self, y_true: float, q_lo: float, q_hi: float) -> dict:
        err = 1.0 if (y_true < q_lo - self.delta) or (y_true > q_hi + self.delta) else 0.0
        self.delta = max(0.0, self.delta + self.gamma * (err - self.alpha_target))
        record = {
            "y": y_true, "q_lo": q_lo - self.delta, "q_hi": q_hi + self.delta,
            "delta": self.delta, "err": err,
        }
        self._history.append(record)
        return record

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)
