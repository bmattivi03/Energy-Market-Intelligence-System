"""Conformal calibration of the quantile forecasts.

Mirrors src/module_b/evaluation.py ConformalQuantileRegressor: split-CQR with the
Barber finite-sample quantile ceil((n+1)(1-alpha))/n. Applied per horizon to the
ENDPOINTS ONLY (q10/q90); the q50 median is never adjusted, to protect the
MAE-scored point forecast. Conformity scores and adjustments are computed on the
original (already inverse-transformed) scale of the prediction frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def barber_level(n: int, alpha: float) -> float:
    """The finite-sample quantile level ceil((n+1)(1-alpha))/n, capped at 1."""
    return float(min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))


def _delta(q_lo, q_hi, y, alpha) -> float:
    scores = np.maximum(q_lo - y, y - q_hi)
    n = len(scores)
    if n == 0:
        return 0.0
    return float(np.quantile(scores, barber_level(n, alpha), method="higher"))


def conformalize(val_pred: pd.DataFrame, val_y: pd.Series, test_pred: pd.DataFrame,
                 alpha: float = 0.20) -> pd.DataFrame:
    """Widen q10/q90 of test_pred using per-horizon deltas from the val set.

    val_pred/test_pred: DataFrames indexed by (origin_ts, horizon_h) with
    q10/q50/q90. val_y: Series aligned to val_pred's index (actual values).
    Returns a calibrated copy of test_pred (q50 unchanged).
    """
    out = test_pred.copy()
    horizons = test_pred.index.get_level_values("horizon_h")
    vh = val_pred.index.get_level_values("horizon_h")
    for h in sorted(set(horizons)):
        vmask = vh == h
        if not vmask.any():
            continue
        delta = _delta(
            val_pred.loc[vmask, "q10"].to_numpy(),
            val_pred.loc[vmask, "q90"].to_numpy(),
            val_y.loc[val_pred.index[vmask]].to_numpy(),
            alpha,
        )
        tmask = horizons == h
        out.loc[tmask, "q10"] = test_pred.loc[tmask, "q10"] - delta
        out.loc[tmask, "q90"] = test_pred.loc[tmask, "q90"] + delta
    return out


def aci_calibrate(val_pred: pd.DataFrame, val_y: pd.Series, test_pred: pd.DataFrame,
                  test_y: pd.Series, alpha: float = 0.20, gamma: float = 0.01) -> pd.DataFrame:
    """Adaptive Conformal Inference (Gibbs-Candes 2021), per horizon, online.

    Reported as a shift-robust ablation: it restores long-run average coverage at
    a width/Winkler cost. q50 unchanged.
    """
    out = test_pred.copy()
    horizons = test_pred.index.get_level_values("horizon_h")
    vh = val_pred.index.get_level_values("horizon_h")
    for h in sorted(set(horizons)):
        vmask = vh == h
        base = _delta(
            val_pred.loc[vmask, "q10"].to_numpy(),
            val_pred.loc[vmask, "q90"].to_numpy(),
            val_y.loc[val_pred.index[vmask]].to_numpy(),
            alpha,
        ) if vmask.any() else 0.0
        tmask = horizons == h
        tp = test_pred.loc[tmask]
        ty = test_y.loc[tp.index].to_numpy()
        lo = tp["q10"].to_numpy(); hi = tp["q90"].to_numpy()
        a_t = alpha
        new_lo, new_hi = [], []
        for i in range(len(tp)):
            d = base * (1.0 + (alpha - a_t) / max(alpha, 1e-6))
            l, u = lo[i] - d, hi[i] + d
            new_lo.append(l); new_hi.append(u)
            covered = (ty[i] >= l) and (ty[i] <= u)
            a_t = a_t + gamma * (alpha - (0.0 if covered else 1.0))
        out.loc[tmask, "q10"] = new_lo
        out.loc[tmask, "q90"] = new_hi
    return out
