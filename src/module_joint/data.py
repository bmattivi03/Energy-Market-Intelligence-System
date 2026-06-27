"""Rolling-origin windowing for the joint model.

Builds, per forecast origin t:
  past:   (lookback, F_past)  scaled fundamentals (price asinh'd) + calendar
  future: (horizon, F_fut)    calendar(target hours) + forecastable weather
                              + Module A load quantiles for h1..h24 (optional)
  targets: load, price (asinh+std), residual_load, renewables, spike

Leakage controls: a window is emitted only if its origin and all 24 target hours
fall inside the requested split index (lookback context may come from before the
split). Future weather uses actuals as a forecast proxy (noise-augmented at train
time in train.py). The only future-load source is Module A's causal export.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import schemas

from .config import HORIZON, LOOKBACK
from .transforms import FeatureScaler, ScalarScaler, asinh

WEATHER_COLS = list(schemas.WEATHER_COLS)
RENEWABLE_COLS = [c for c in schemas.RENEWABLE_COLS]
WIND_COLS = [c for c in ("gen_wind_onshore", "gen_wind_offshore")]
SOLAR_COL = "gen_solar"
CAL_DIM = 8  # hour/dow/month sin+cos, is_weekend, is_holiday

try:  # holidays is optional; fall back to weekend-only if absent
    import holidays as _holidays_lib

    _DE_HOLIDAYS = _holidays_lib.Germany()
except Exception:  # pragma: no cover - environment dependent
    _DE_HOLIDAYS = None


def calendar_features(idx: pd.DatetimeIndex) -> np.ndarray:
    """(len(idx), CAL_DIM) leak-free calendar encodings."""
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    month = (idx.month.to_numpy() - 1)
    feats = np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            (dow >= 5).astype(float),
            _holiday_flags(idx),
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def _holiday_flags(idx: pd.DatetimeIndex) -> np.ndarray:
    if _DE_HOLIDAYS is None:
        return np.zeros(len(idx), dtype=float)
    return np.array([1.0 if d.date() in _DE_HOLIDAYS else 0.0 for d in idx])


def spike_threshold(price: pd.Series, min_periods: int = 720, q: float = 0.9) -> pd.Series:
    """Causal expanding 90th percentile, matching Module B's spike feature."""
    return price.shift(1).expanding(min_periods=min_periods).quantile(q)


# Engineered fundamental features reused from Module B (feature parity with CatBoost).
ENGINEERED_BUNDLES = ["lags", "fundamentals", "spike", "regime"]


def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment df with Module B's engineered features (causal). Lazy-imports
    module_b AFTER torch is loaded (torch comes in via .transforms) to avoid the
    dual-OpenMP segfault. Warmup NaNs are handled downstream by nan_to_num.
    """
    import module_b.features as F  # lazy: torch already imported via transforms

    return F.build_features(df, ENGINEERED_BUNDLES)


@dataclass
class WindowArrays:
    past: np.ndarray  # (N, lookback, F_past)
    future: np.ndarray  # (N, horizon, F_fut)
    y: dict  # scaled targets: load/price (N,H); residual_load (N,H); renewables (N,H,2); spike (N,H)
    y_raw: dict  # original-scale load (MW) and price (EUR/MWh), (N,H)
    origin_ts: np.ndarray  # (N,) origin timestamps
    anchor: np.ndarray | None = None  # (N,) fuel-cost anchor at origin (raw EUR/MWh)


@dataclass
class Scalers:
    x: FeatureScaler
    load: ScalarScaler
    price: ScalarScaler  # fit on asinh(price), or asinh(price - anchor) if anchor_residual
    residual: ScalarScaler
    renew: FeatureScaler
    past_cols: list
    price_idx: int
    weather_idx: list
    use_load_quantiles: bool
    anchor_residual: bool = False
    anchor_col: str = "clean_spark_anchor"


def _renewable_sum(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in RENEWABLE_COLS if c in df.columns]
    return df[cols].sum(axis=1) if cols else pd.Series(0.0, index=df.index)


def _wind_solar(df: pd.DataFrame) -> np.ndarray:
    wind = df[[c for c in WIND_COLS if c in df.columns]].sum(axis=1).to_numpy()
    solar = df[SOLAR_COL].to_numpy() if SOLAR_COL in df.columns else np.zeros(len(df))
    return np.stack([wind, solar], axis=1)


def fit_scalers(df: pd.DataFrame, use_load_quantiles: bool,
                anchor_residual: bool = False,
                anchor_col: str = "clean_spark_anchor") -> Scalers:
    past_cols = list(df.columns)
    price_idx = past_cols.index(schemas.PRICE_COL)
    weather_idx = [past_cols.index(c) for c in WEATHER_COLS if c in past_cols]

    P = df[past_cols].to_numpy(dtype=float).copy()
    P[:, price_idx] = asinh(P[:, price_idx])
    x = FeatureScaler().fit(P)

    load = ScalarScaler().fit(df[schemas.LOAD_COL].to_numpy())
    if anchor_residual:
        resid = df[schemas.PRICE_COL].to_numpy() - df[anchor_col].to_numpy()
        price = ScalarScaler().fit(asinh(resid))
    else:
        price = ScalarScaler().fit(asinh(df[schemas.PRICE_COL].to_numpy()))
    residual = ScalarScaler().fit(_renewable_sum(df).rsub(df[schemas.LOAD_COL]).to_numpy())
    renew = FeatureScaler().fit(_wind_solar(df))
    return Scalers(x, load, price, residual, renew, past_cols, price_idx,
                   weather_idx, use_load_quantiles, anchor_residual, anchor_col)


def build_windows(
    df: pd.DataFrame,
    *,
    load_quantiles: pd.DataFrame | None,
    scalers: Scalers | None = None,
    fit: bool = False,
    use_load_quantiles: bool = True,
    restrict_to: pd.DatetimeIndex | None = None,
    fundamentals: bool = False,
    anchor_residual: bool = False,
    anchor_col: str = "clean_spark_anchor",
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[WindowArrays, Scalers]:
    """Build rolling-origin windows. Pass fit=True on the train split to fit the
    scalers; reuse the returned scalers for val/test. With fundamentals=True the
    df is augmented with Module B's engineered features (feature parity). With
    anchor_residual=True the price target is modelled as price - fuel-cost anchor.
    """
    if fundamentals or anchor_residual:
        df = engineered_features(df)

    if scalers is None:
        if not fit:
            raise ValueError("scalers must be provided when fit=False")
        scalers = fit_scalers(df, use_load_quantiles, anchor_residual, anchor_col)

    past_cols = scalers.past_cols
    P = df[past_cols].to_numpy(dtype=float).copy()
    P[:, scalers.price_idx] = asinh(P[:, scalers.price_idx])
    P = scalers.x.transform(P).astype(np.float32)
    P = np.nan_to_num(P, nan=0.0)  # engineered-feature warmup NaNs -> mean (0 after scaling)

    cal_all = calendar_features(df.index)  # (T, CAL_DIM)
    weather_scaled = (
        (df[WEATHER_COLS].to_numpy(dtype=float) - scalers.x.mean_[scalers.weather_idx])
        / scalers.x.std_[scalers.weather_idx]
    ).astype(np.float32)

    load_raw = df[schemas.LOAD_COL].to_numpy(dtype=float)
    price_raw = df[schemas.PRICE_COL].to_numpy(dtype=float)
    residual_raw = (df[schemas.LOAD_COL] - _renewable_sum(df)).to_numpy(dtype=float)
    windsolar_raw = _wind_solar(df)
    thr = spike_threshold(df[schemas.PRICE_COL]).to_numpy(dtype=float)
    thr = np.where(np.isnan(thr), np.inf, thr)

    T = len(df)
    lq = load_quantiles
    use_lq = lq is not None and use_load_quantiles

    # --- candidate origins p in [lookback-1, T-horizon-1] filtered vectorially ---
    positions = np.arange(lookback - 1, T - horizon)
    mask = np.ones(len(positions), dtype=bool)
    if restrict_to is not None:
        in_r = df.index.isin(restrict_to)  # (T,)
        mask &= in_r[positions]  # origin in split
        mask &= in_r[positions + horizon]  # last target in split (contiguous splits)
    if use_lq:
        mask &= df.index.isin(lq.index)[positions]  # origin has a Module A forecast
    p_arr = positions[mask]

    # --- past windows via sliding view: window s covers rows [s, s+lookback-1] ---
    past_full = np.concatenate([P, cal_all], axis=1)  # (T, F+CAL)
    past_sw = np.lib.stride_tricks.sliding_window_view(past_full, lookback, axis=0)
    past = np.ascontiguousarray(past_sw[p_arr - lookback + 1].transpose(0, 2, 1))  # (N,L,C)

    # --- future cal+weather: window starting at target_start = p+1 ---
    fut_src = np.concatenate([cal_all, weather_scaled], axis=1)  # (T, CAL+W)
    fut_sw = np.lib.stride_tricks.sliding_window_view(fut_src, horizon, axis=0)
    fut_cw = fut_sw[p_arr + 1].transpose(0, 2, 1)  # (N,H,CAL+W)

    if use_lq:
        cols = [f"load_q{q}_h{h}" for q in (10, 50, 90) for h in range(1, horizon + 1)]
        lq_arr = lq.reindex(df.index)[cols].to_numpy(dtype=float)  # (T,72) q-major
        lq_arr = scalers.load.transform(lq_arr)  # MW -> standardized via load scaler
        lq_block = lq_arr.reshape(T, 3, horizon).transpose(0, 2, 1)  # (T,H,3) [q10,q50,q90]
        future = np.concatenate([fut_cw, lq_block[p_arr]], axis=2).astype(np.float32)
    else:
        future = fut_cw.astype(np.float32)

    # --- targets: window of horizon starting at p+1 ---
    def _tw(arr):  # (T,...) -> windows (T-h+1, h, ...) then pick p+1
        sw = np.lib.stride_tricks.sliding_window_view(arr, horizon, axis=0)
        if arr.ndim == 1:
            return sw[p_arr + 1]  # (N,H)
        return sw[p_arr + 1].transpose(0, 2, 1)  # (N,H,C)

    load_w = _tw(load_raw)
    price_w = _tw(price_raw)
    resid_w = _tw(residual_raw)
    renew_w = _tw(windsolar_raw)  # (N,H,2)
    thr_w = _tw(thr)

    # price target: residual to the fuel-cost anchor (anchor at origin), or asinh(price)
    anchor_origin = None
    if scalers.anchor_residual:
        anchor_all = df[scalers.anchor_col].to_numpy(dtype=float)
        anchor_origin = anchor_all[p_arr]  # (N,) raw EUR/MWh
        price_target = price_w - anchor_origin[:, None]
    else:
        price_target = price_w
    y_price = scalers.price.transform(asinh(price_target)).astype(np.float32)

    wa = WindowArrays(
        past=past,
        future=future,
        y={
            "load": scalers.load.transform(load_w).astype(np.float32),
            "price": y_price,
            "residual_load": scalers.residual.transform(resid_w).astype(np.float32),
            "renewables": scalers.renew.transform(renew_w).astype(np.float32),
            "spike": (price_w > thr_w).astype(np.float32),
        },
        y_raw={
            "load": load_w.astype(np.float32),
            "price": price_w.astype(np.float32),
        },
        origin_ts=df.index[p_arr].to_numpy(),
        anchor=anchor_origin,
    )
    return wa, scalers
