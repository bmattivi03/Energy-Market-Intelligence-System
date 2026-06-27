"""Evaluation adapters: compare the joint model against the Module A/B baselines
using the shared metrics in src/module_b/evaluation.py, broken down by horizon
segment, with Diebold-Mariano tests and a DM power calculation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from module_b import evaluation as ev

SEGMENTS = {"h1_6": range(1, 7), "h7_18": range(7, 19), "h19_24": range(19, 25)}
QUANTILES = (0.1, 0.5, 0.9)


def make_truth(df: pd.DataFrame, index: pd.MultiIndex, col: str) -> pd.Series:
    """Actual values aligned to a (origin_ts, horizon_h) prediction index."""
    origins = index.get_level_values("origin_ts")
    horizons = index.get_level_values("horizon_h")
    target_ts = origins + pd.to_timedelta(horizons, unit="h")
    vals = df[col].reindex(target_ts).to_numpy()
    return pd.Series(vals, index=index)


def _metrics(pred: pd.DataFrame, truth: pd.Series) -> dict:
    y = truth.to_numpy()
    q50 = pred["q50"].to_numpy()
    return {
        "mae": ev.mae(y, q50),
        "rmse": ev.rmse(y, q50),
        "pinball": ev.multi_pinball_loss(y, pred, QUANTILES),
        "coverage": ev.coverage(y, pred["q10"].to_numpy(), pred["q90"].to_numpy()),
        "winkler": ev.winkler_score(y, pred["q10"].to_numpy(), pred["q90"].to_numpy(), 0.20),
        "n": int(len(y)),
    }


def _by_segment(pred: pd.DataFrame, truth: pd.Series) -> dict:
    h = pred.index.get_level_values("horizon_h")
    seg = {}
    for name, rng in SEGMENTS.items():
        mask = h.isin(list(rng))
        if mask.any():
            seg[name] = _metrics(pred.loc[mask], truth.loc[pred.index[mask]])
    return seg


def _compare(pred, truth, baseline_pred):
    res = {"overall": _metrics(pred, truth), "segments": _by_segment(pred, truth)}
    if baseline_pred is not None:
        common = pred.index.intersection(baseline_pred.index)
        y = truth.loc[common].to_numpy()
        e_model = np.abs(y - pred.loc[common, "q50"].to_numpy())
        e_base = np.abs(y - baseline_pred.loc[common, "q50"].to_numpy())
        dm = ev.diebold_mariano(e_model, e_base)
        res["dm_vs_baseline"] = {"statistic": dm.statistic, "p_value": dm.p_value}
    return res


def compare_price(pred, truth, baseline_pred=None):
    return _compare(pred, truth, baseline_pred)


def compare_load(pred, truth, baseline_pred=None):
    return _compare(pred, truth, baseline_pred)


def dm_power(n: int, sd_loss_diff: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable mean loss difference for a two-sided DM test."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * sd_loss_diff / np.sqrt(max(n, 1)))


def leaderboard(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
