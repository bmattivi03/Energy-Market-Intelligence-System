"""Feature engineering for DE-LU load forecasting (Module A).

Each ``add_*`` function appends new columns to a copy of the input frame.
``REGISTRY`` composes them in dependency order; ``build_features`` applies
a selected subset.

Groups:

* :func:`add_calendar`   — cyclical hour/dow/month, weekend, DE holidays, DST.
* :func:`add_load_lags`  — load lags + rolling stats (primary autoregressive signal).
* :func:`add_weather`    — lagged weather + degree-hours + cross-city aggregates.
* :func:`add_renewables` — lagged wind/solar totals + residual load proxy.

No leakage guarantee: at origin t, only data through t-1h enters any feature.
The raw weather columns in the lookback window are available to the LSTM
sequence encoder; ``add_weather`` adds derived features on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from data.schemas import CITY_NAMES, LOAD_COL, WEATHER_VARIABLES
from data.splits import HORIZON

# ---------------------------------------------------------------- constants

TARGET_COL = LOAD_COL
ORIGIN_COL = "origin_ts"
HORIZON_COL = "horizon_h"

DEFAULT_LOAD_LAGS: tuple[int, ...] = (24, 48, 168, 336)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (24, 168)

WIND_COLS = ("gen_wind_onshore", "gen_wind_offshore")
SOLAR_COL = "gen_solar"

# Heating/cooling thresholds (°C) — standard German building-physics values
HEATING_BASE = 15.0
COOLING_BASE = 22.0


# ---------------------------------------------------------------- calendar


def _de_holidays_set(years: range) -> set:
    import holidays as hol
    return set(hol.country_holidays("DE", years=list(years)).keys())


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time encodings, weekend, DE holidays, DST flag."""
    out = df.copy()
    idx = out.index
    out["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24).astype(np.float32)
    out["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24).astype(np.float32)
    out["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7).astype(np.float32)
    out["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7).astype(np.float32)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12).astype(np.float32)
    out["is_weekend"] = (idx.dayofweek >= 5).astype(np.float32)

    holiday_dates = _de_holidays_set(range(idx.year.min(), idx.year.max() + 1))
    out["is_holiday_DE"] = pd.Series(
        [float(d.date() in holiday_dates) for d in idx], index=idx, dtype=np.float32
    )
    if idx.tz is not None:
        local = idx.tz_convert("Europe/Berlin")
        deltas = local.to_series().diff().dt.total_seconds().fillna(3600).astype(int)
        out["is_dst_transition"] = (deltas != 3600).astype(np.float32).values
    else:
        out["is_dst_transition"] = np.zeros(len(idx), dtype=np.float32)
    return out


# ---------------------------------------------------------------- load lags


def add_load_lags(
    df: pd.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_LOAD_LAGS,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Load lags and rolling statistics.

    Lags at 24h/48h/168h/336h capture same-time-yesterday, 2-days-ago,
    same-time-last-week, and same-time-two-weeks-ago.
    Rolling stats are shifted by 1h so t-1h is the most recent observation.
    """
    out = df.copy()
    s = out[LOAD_COL]
    for lag in lags:
        out[f"load_lag{lag}h"] = s.shift(lag).astype(np.float32)
    for w in windows:
        rolling = s.rolling(w, min_periods=max(2, w // 4))
        out[f"load_rmean{w}h"] = rolling.mean().shift(1).astype(np.float32)
        out[f"load_rstd{w}h"]  = rolling.std().shift(1).astype(np.float32)
        out[f"load_rmin{w}h"]  = rolling.min().shift(1).astype(np.float32)
        out[f"load_rmax{w}h"]  = rolling.max().shift(1).astype(np.float32)
    return out


# ---------------------------------------------------------------- weather


def _weather_cols(cities: list[str] | None = None) -> list[str]:
    cities = cities or list(CITY_NAMES)
    return [f"{c}_{v}" for c in cities for v in WEATHER_VARIABLES]


def add_weather(
    df: pd.DataFrame,
    *,
    temp_lags: tuple[int, ...] = (24, 48, 168),
    other_lags: tuple[int, ...] = (24,),
) -> pd.DataFrame:
    """Lagged weather + cross-city aggregates + degree-hour features.

    Temperature lags matter most for load (heating/cooling demand).
    Wind/radiation lags add generation-mix context.
    Degree-hours encode non-linear relationship between temp and load.
    """
    out = df.copy()
    cities = list(CITY_NAMES)
    temp_cols    = [f"{c}_temperature_2m"       for c in cities]
    wind10_cols  = [f"{c}_wind_speed_10m"       for c in cities]
    wind100_cols = [f"{c}_wind_speed_100m"      for c in cities]
    rad_cols     = [f"{c}_shortwave_radiation"  for c in cities]

    # Lagged temperature (most predictive for load)
    for col in temp_cols:
        if col not in out.columns:
            continue
        for lag in temp_lags:
            out[f"{col}_lag{lag}h"] = out[col].shift(lag).astype(np.float32)

    # Lagged wind + radiation
    for cols, lags in [(wind10_cols + wind100_cols + rad_cols, other_lags)]:
        for col in cols:
            if col not in out.columns:
                continue
            for lag in lags:
                out[f"{col}_lag{lag}h"] = out[col].shift(lag).astype(np.float32)

    # Cross-city mean temperature (lag 24h — main exogenous signal)
    present_temp = [c for c in temp_cols if c in out.columns]
    if present_temp:
        mean_temp = out[present_temp].mean(axis=1)
        out["weather_mean_temp"] = mean_temp.astype(np.float32)
        out["weather_mean_temp_lag24h"] = mean_temp.shift(24).astype(np.float32)

        # Heating / cooling degree-hours: capture non-linear temp→load relationship
        out["heating_dh"] = mean_temp.clip(upper=HEATING_BASE).rsub(HEATING_BASE).astype(np.float32)
        out["cooling_dh"] = mean_temp.clip(lower=COOLING_BASE).sub(COOLING_BASE).astype(np.float32)
        out["heating_dh_lag24h"] = out["heating_dh"].shift(24).astype(np.float32)
        out["cooling_dh_lag24h"] = out["cooling_dh"].shift(24).astype(np.float32)

    # Cross-city mean wind speed (lag 24h — wind-chill proxy)
    present_wind = [c for c in wind10_cols if c in out.columns]
    if present_wind:
        out["weather_mean_wind_lag24h"] = (
            out[present_wind].mean(axis=1).shift(24).astype(np.float32)
        )

    # Cross-city mean solar radiation (lag 24h — daylight proxy)
    present_rad = [c for c in rad_cols if c in out.columns]
    if present_rad:
        out["weather_mean_rad_lag24h"] = (
            out[present_rad].mean(axis=1).shift(24).astype(np.float32)
        )

    return out


# ---------------------------------------------------------------- renewables


def add_renewables(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged renewable generation and residual-load proxy.

    Wind + solar at lag 24h give the model context about supply conditions
    that correlate with net load and cross-border flows.
    """
    out = df.copy()

    wind_present = [c for c in WIND_COLS if c in out.columns]
    solar_present = SOLAR_COL if SOLAR_COL in out.columns else None

    if wind_present:
        wind_total = sum(out[c] for c in wind_present)
        out["wind_total_lag24h"] = wind_total.shift(24).astype(np.float32)
        out["wind_total_lag168h"] = wind_total.shift(168).astype(np.float32)

    if solar_present:
        out["solar_lag24h"]  = out[SOLAR_COL].shift(24).astype(np.float32)
        out["solar_lag168h"] = out[SOLAR_COL].shift(168).astype(np.float32)

    if wind_present and solar_present:
        renewable = wind_total + out[SOLAR_COL]
        load_lag  = out[LOAD_COL].shift(24)
        out["residual_load_lag24h"] = (load_lag - renewable.shift(24)).astype(np.float32)

    return out


# ============================================================ registry

BundleFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class _BundleSpec:
    fn: BundleFn
    requires: tuple[str, ...] = ()


class _Registry(dict):
    def resolve_order(self, names: Iterable[str]) -> list[_BundleSpec]:
        wanted = list(dict.fromkeys(names))
        ordered: list[_BundleSpec] = []
        seen: set[str] = set()

        def visit(n: str, stack: tuple[str, ...]) -> None:
            if n in seen:
                return
            if n in stack:
                raise ValueError(f"Cycle in feature bundles: {stack + (n,)}")
            if n not in self:
                raise KeyError(f"Unknown bundle: {n!r}; have {list(self)}")
            spec = self[n]
            for req in spec.requires:
                visit(req, stack + (n,))
            seen.add(n)
            ordered.append(spec)

        for n in wanted:
            visit(n, ())
        return ordered


REGISTRY: _Registry = _Registry({
    "calendar":   _BundleSpec(add_calendar),
    "load_lags":  _BundleSpec(add_load_lags),
    "weather":    _BundleSpec(add_weather),
    "renewables": _BundleSpec(add_renewables),
})

ALL_BUNDLES: tuple[str, ...] = ("calendar", "load_lags", "weather", "renewables")


def build_features(
    df: pd.DataFrame,
    bundles: Sequence[str] = ALL_BUNDLES,
) -> pd.DataFrame:
    """Apply selected feature bundles in dependency order. Returns copy."""
    out = df
    for spec in REGISTRY.resolve_order(bundles):
        out = spec.fn(out)
    return out


# ============================================================ feature column lists
# Populated after build_features — used by train.py to slice feature arrays.

CALENDAR_COLS: tuple[str, ...] = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday_DE", "is_dst_transition",
)

LOAD_LAG_COLS: tuple[str, ...] = (
    "load_lag24h", "load_lag48h", "load_lag168h", "load_lag336h",
    "load_rmean24h", "load_rstd24h", "load_rmin24h", "load_rmax24h",
    "load_rmean168h", "load_rstd168h", "load_rmin168h", "load_rmax168h",
)

WEATHER_DERIVED_COLS: tuple[str, ...] = (
    "weather_mean_temp", "weather_mean_temp_lag24h",
    "heating_dh", "cooling_dh", "heating_dh_lag24h", "cooling_dh_lag24h",
    "weather_mean_wind_lag24h", "weather_mean_rad_lag24h",
)

RENEWABLE_COLS: tuple[str, ...] = (
    "wind_total_lag24h", "wind_total_lag168h",
    "solar_lag24h", "solar_lag168h",
    "residual_load_lag24h",
)
