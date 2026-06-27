"""Evaluation suite for Module A load forecasts.

Metrics
-------
Point (q50):  MAE, RMSE, MAPE, nMAE
Probabilistic: pinball loss per quantile, interval coverage, mean width, Winkler score
Baselines:    naive (lag-24h), seasonal-naive (lag-168h)
Segments:     hour-of-day group, weekday/weekend, season, horizon group
Calibration:  empirical coverage at each quantile level
Significance: Diebold-Mariano test vs each baseline
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from data.splits import HORIZON
from module_a.features import TARGET_COL

QUANTILES = (0.1, 0.5, 0.9)
Q_COLS    = ("q10", "q50", "q90")

# ---------------------------------------------------------------- helpers


def _arr(*xs) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(x, dtype=np.float64) for x in xs)


# ---------------------------------------------------------------- point metrics


def mae(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    yt, yp = _arr(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), eps)))


def nmae(y_true, y_pred) -> float:
    """MAE normalised by mean of actuals (unit-free %)."""
    yt, yp = _arr(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt)))


# ---------------------------------------------------------------- probabilistic metrics


def pinball(y_true, y_pred, q: float) -> float:
    yt, yp = _arr(y_true, y_pred)
    err = yt - yp
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))


def coverage(y_true, q_low, q_high) -> float:
    """Fraction of actuals inside [q_low, q_high]."""
    yt, lo, hi = _arr(y_true, q_low, q_high)
    return float(np.mean((yt >= lo) & (yt <= hi)))


def mean_interval_width(q_low, q_high) -> float:
    lo, hi = _arr(q_low, q_high)
    return float(np.mean(hi - lo))


def winkler_score(y_true, q_low, q_high, alpha: float = 0.20) -> float:
    """Winkler score for (1-alpha) prediction interval. Lower = better."""
    yt, lo, hi = _arr(y_true, q_low, q_high)
    width = hi - lo
    below = (yt < lo).astype(float)
    above = (yt > hi).astype(float)
    penalty = (2 / alpha) * (below * (lo - yt) + above * (yt - hi))
    return float(np.mean(width + penalty))


def empirical_coverage_at_quantiles(
    y_true, q10_pred, q50_pred, q90_pred
) -> dict[str, float]:
    """Fraction of actuals below each predicted quantile (should match q level)."""
    yt, q10, q50, q90 = _arr(y_true, q10_pred, q50_pred, q90_pred)
    return {
        "below_q10": float(np.mean(yt < q10)),
        "below_q50": float(np.mean(yt < q50)),
        "below_q90": float(np.mean(yt < q90)),
    }


# ---------------------------------------------------------------- baselines


def naive_forecast(load_series: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    """ŷ_{t+h} = load_{t+h-24}  (same hour yesterday)."""
    rows = []
    for t in load_series.index:
        for h in range(1, horizon + 1):
            anchor_ts = t + pd.Timedelta(hours=h - 24)
            if anchor_ts in load_series.index:
                rows.append({"origin_ts": t, "horizon_h": h,
                             "pred": load_series[anchor_ts]})
    return pd.DataFrame(rows).set_index(["origin_ts", "horizon_h"])


def seasonal_naive_forecast(load_series: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    """ŷ_{t+h} = load_{t+h-168}  (same hour last week)."""
    rows = []
    for t in load_series.index:
        for h in range(1, horizon + 1):
            anchor_ts = t + pd.Timedelta(hours=h - 168)
            if anchor_ts in load_series.index:
                rows.append({"origin_ts": t, "horizon_h": h,
                             "pred": load_series[anchor_ts]})
    return pd.DataFrame(rows).set_index(["origin_ts", "horizon_h"])


def _build_baseline_fast(load_series: pd.Series, lag_hours: int) -> pd.Series:
    """Vectorised baseline: shift load by lag_hours, align to (origin+h) targets."""
    return load_series.shift(lag_hours)


# ---------------------------------------------------------------- Diebold-Mariano


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    h: int = 1,
) -> tuple[float, float]:
    """Two-sided DM test. Returns (DM statistic, p-value).

    errors_a, errors_b: squared or absolute forecast errors (same length).
    H0: equal predictive accuracy.
    """
    d = errors_a - errors_b
    n = len(d)
    d_bar = d.mean()
    # Newey-West variance with h-1 lags
    gamma0 = np.var(d, ddof=0)
    gamma = sum(
        (1 - k / h) * np.cov(d[k:], d[:-k])[0, 1]
        for k in range(1, h)
    ) if h > 1 else 0.0
    var_d = (gamma0 + 2 * gamma) / n
    if var_d <= 0:
        return 0.0, 1.0
    dm_stat = d_bar / np.sqrt(var_d)
    p_val = 2 * stats.norm.sf(abs(dm_stat))
    return float(dm_stat), float(p_val)


# ---------------------------------------------------------------- segmentation


def _season(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"}[month]


def _hour_group(hour: int) -> str:
    if 6 <= hour <= 9:   return "morning_peak"
    if 10 <= hour <= 16: return "midday"
    if 17 <= hour <= 20: return "evening_peak"
    return "night"


SEGMENT_FNS: dict[str, Callable[[pd.Timestamp], str]] = {
    "day_type":   lambda ts: "weekend" if ts.dayofweek >= 5 else "weekday",
    "season":     lambda ts: _season(ts.month),
    "hour_group": lambda ts: _hour_group(ts.hour),
}


# ---------------------------------------------------------------- full report


def point_metrics_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "model",
) -> pd.DataFrame:
    return pd.DataFrame([{
        "model":  label,
        "MAE":    mae(y_true, y_pred),
        "RMSE":   rmse(y_true, y_pred),
        "MAPE%":  round(mape(y_true, y_pred) * 100, 3),
        "nMAE%":  round(nmae(y_true, y_pred) * 100, 3),
    }])


def probabilistic_metrics_table(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    label: str = "model",
) -> pd.DataFrame:
    cov = empirical_coverage_at_quantiles(y_true, q10, q50, q90)
    return pd.DataFrame([{
        "model":         label,
        "pinball_q10":   round(pinball(y_true, q10, 0.1), 2),
        "pinball_q50":   round(pinball(y_true, q50, 0.5), 2),
        "pinball_q90":   round(pinball(y_true, q90, 0.9), 2),
        "coverage_80%":  round(coverage(y_true, q10, q90) * 100, 1),
        "width_MW":      round(mean_interval_width(q10, q90), 0),
        "winkler":       round(winkler_score(y_true, q10, q90), 2),
        "calib_q10":     round(cov["below_q10"] * 100, 1),
        "calib_q50":     round(cov["below_q50"] * 100, 1),
        "calib_q90":     round(cov["below_q90"] * 100, 1),
    }])


def segment_report(
    long_preds: pd.DataFrame,
    actual: pd.Series,
    segment_fn: Callable[[pd.Timestamp], str],
    segment_name: str,
) -> pd.DataFrame:
    """MAE and pinball_q50 broken down by segment."""
    rows = []
    for (origin, h), pred_row in long_preds.iterrows():
        target_ts = origin + pd.Timedelta(hours=int(h))
        if target_ts not in actual.index:
            continue
        y = actual[target_ts]
        seg = segment_fn(origin)
        rows.append({
            segment_name: seg,
            "actual":     y,
            "q50_pred":   pred_row["q50"],
            "q10_pred":   pred_row["q10"],
            "q90_pred":   pred_row["q90"],
        })
    df = pd.DataFrame(rows)
    return (
        df.groupby(segment_name)
        .apply(lambda g: pd.Series({
            "MAE":         mae(g["actual"], g["q50_pred"]),
            "RMSE":        rmse(g["actual"], g["q50_pred"]),
            "nMAE%":       round(nmae(g["actual"], g["q50_pred"]) * 100, 2),
            "coverage_80": round(coverage(g["actual"], g["q10_pred"], g["q90_pred"]) * 100, 1),
            "n":           len(g),
        }), include_groups=False)
        .reset_index()
    )


def full_report(
    long_preds: pd.DataFrame,
    actual: pd.Series,
    split_name: str = "val",
    naive_lag: int = 24,
    seasonal_lag: int = 168,
) -> None:
    """Print a complete evaluation report to stdout."""
    print(f"\n{'='*60}")
    print(f"Module A Evaluation — {split_name}")
    print(f"{'='*60}")

    # Align predictions to actuals
    pairs = []
    for (origin, h), pred_row in long_preds.iterrows():
        target_ts = origin + pd.Timedelta(hours=int(h))
        if target_ts not in actual.index:
            continue
        pairs.append({
            "target_ts": target_ts,
            "actual":    actual[target_ts],
            "q10":       pred_row["q10"],
            "q50":       pred_row["q50"],
            "q90":       pred_row["q90"],
            "h":         int(h),
        })
    df = pd.DataFrame(pairs)
    if df.empty:
        print("  No overlapping predictions and actuals found.")
        return

    yt = df["actual"].values
    q10, q50, q90 = df["q10"].values, df["q50"].values, df["q90"].values

    # --- point metrics
    naive_pred   = actual.shift(naive_lag).reindex(df["target_ts"]).values
    seasonal_pred = actual.shift(seasonal_lag).reindex(df["target_ts"]).values

    mask = ~np.isnan(naive_pred) & ~np.isnan(seasonal_pred)

    print("\n--- Point metrics (q50) ---")
    pt = pd.concat([
        point_metrics_table(yt[mask], q50[mask],            "MultiScaleLSTM"),
        point_metrics_table(yt[mask], naive_pred[mask],     f"Naive (lag-{naive_lag}h)"),
        point_metrics_table(yt[mask], seasonal_pred[mask],  f"SeasonalNaive (lag-{seasonal_lag}h)"),
    ], ignore_index=True).set_index("model")
    print(pt.to_string())

    # --- probabilistic metrics
    print("\n--- Probabilistic metrics ---")
    prob = probabilistic_metrics_table(yt, q10, q50, q90, "MultiScaleLSTM")
    print(prob.set_index("model").to_string())

    # --- DM tests
    print("\n--- Diebold-Mariano vs baselines (absolute errors, q50) ---")
    errs_model    = np.abs(yt[mask] - q50[mask])
    errs_naive    = np.abs(yt[mask] - naive_pred[mask])
    errs_seasonal = np.abs(yt[mask] - seasonal_pred[mask])
    dm_stat, dm_p = diebold_mariano(errs_model, errs_naive)
    print(f"  vs Naive        DM={dm_stat:+.3f}  p={dm_p:.4f}{'  *' if dm_p < 0.05 else ''}")
    dm_stat, dm_p = diebold_mariano(errs_model, errs_seasonal)
    print(f"  vs SeasonalNaive DM={dm_stat:+.3f}  p={dm_p:.4f}{'  *' if dm_p < 0.05 else ''}")
    print("  (negative DM = model better; * = significant at 5%)")

    # --- segment analysis
    lp = long_preds.copy()
    for seg_name, seg_fn in SEGMENT_FNS.items():
        print(f"\n--- Segment: {seg_name} ---")
        seg_df = segment_report(lp, actual, seg_fn, seg_name)
        print(seg_df.set_index(seg_name).to_string())

    # --- by horizon group
    print("\n--- By horizon group ---")
    df["h_group"] = pd.cut(
        df["h"], bins=[0, 6, 18, 24],
        labels=["h1-6", "h7-18", "h19-24"],
    )
    by_h = (
        df.groupby("h_group", observed=True)
        .apply(lambda g: pd.Series({
            "MAE":         mae(g["actual"], g["q50"]),
            "nMAE%":       round(nmae(g["actual"], g["q50"]) * 100, 2),
            "coverage_80": round(coverage(g["actual"], g["q10"], g["q90"]) * 100, 1),
            "n":           len(g),
        }), include_groups=False)
    )
    print(by_h.to_string())

    print(f"\n{'='*60}\n")
