"""Feature engineering for day-ahead price forecasting.

Each ``add_*`` function is a stateless transform that appends new columns to
a copy of the input frame. The ``REGISTRY`` composes them in dependency order
so notebooks can pick a bundle list and call :func:`build_features`.

Groups:

* :func:`add_calendar` - cyclical hour/dow/month, weekend, DE holidays, DST.
* :func:`add_price_lags` - standard EPF lags + rolling mean/std/min/max.
* :func:`add_fundamentals` - clean spark/dark spreads, residual load,
  renewable penetration.
* :func:`add_spike` - high-residual-load and renewable-scarcity flags.
* :func:`add_regime` - 2022-2023 European energy-crisis indicator.
* :func:`add_weather` - lagged + aggregated weather features from 5 cities.
* :func:`add_load_quantiles` - Module A load q10/q50/q90 forecasts (optional).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from data.schemas import CITY_NAMES, LOAD_COL, PRICE_COL, WEATHER_VARIABLES
from data.splits import HORIZON


# ---------------------------------------------------------------- calendar

CALENDAR_COLS = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday_DE", "is_dst_transition",
)


def _de_holidays_set(years: range) -> set:
    import holidays as hol
    return set(hol.country_holidays("DE", years=list(years)).keys())


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Append calendar features. Operates on a copy."""
    out = df.copy()
    idx = out.index
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
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


# ---------------------------------------------------------------- price lags

DEFAULT_PRICE_LAGS: tuple[int, ...] = (1, 2, 3, 24, 25, 48, 49, 168, 169)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (24, 168, 336)


def add_price_lags(
    df: pd.DataFrame,
    *,
    column: str = PRICE_COL,
    lags: tuple[int, ...] = DEFAULT_PRICE_LAGS,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Append price lag and rolling features. Operates on a copy."""
    out = df.copy()
    s = out[column]
    for lag in lags:
        out[f"{column}_lag{lag}h"] = s.shift(lag).astype(np.float32)
    for w in windows:
        rolling = s.rolling(w, min_periods=max(2, w // 4))
        out[f"{column}_rmean{w}h"] = rolling.mean().shift(1).astype(np.float32)
        out[f"{column}_rstd{w}h"] = rolling.std().shift(1).astype(np.float32)
        out[f"{column}_rmin{w}h"] = rolling.min().shift(1).astype(np.float32)
        out[f"{column}_rmax{w}h"] = rolling.max().shift(1).astype(np.float32)
    return out


# ---------------------------------------------------------------- fundamentals

GAS_HEAT_RATE = 1.95
GAS_EMISSION_FACTOR = 0.20  # tCO2 / MWh fuel
COAL_HEAT_RATE = 2.46
COAL_EMISSION_FACTOR = 0.34

RUN_OF_RIVER_COL = "gen_hydro_ror"
WIND_COLS = ("gen_wind_onshore", "gen_wind_offshore")
SOLAR_COL = "gen_solar"


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def add_fundamentals(
    df: pd.DataFrame,
    *,
    gas_col: str = "ttf_gas",
    carbon_col: str = "carbon_ets",
    load_col: str = LOAD_COL,
) -> pd.DataFrame:
    """Append clean-spread, residual-load and penetration features."""
    out = df.copy()

    if gas_col in out and carbon_col in out:
        out["clean_spark_anchor"] = (
            out[gas_col] * GAS_HEAT_RATE
            + out[carbon_col].fillna(0) * GAS_EMISSION_FACTOR * GAS_HEAT_RATE
        ).astype(np.float32)
        out["clean_dark_anchor"] = (
            out[gas_col] * COAL_HEAT_RATE
            + out[carbon_col].fillna(0) * COAL_EMISSION_FACTOR * COAL_HEAT_RATE
        ).astype(np.float32)
        out["gas_carbon_interaction"] = (
            out[gas_col] * out[carbon_col].fillna(0)
        ).astype(np.float32)

    if load_col in out:
        wind = sum(out[c] for c in WIND_COLS if c in out)
        solar = out[SOLAR_COL] if SOLAR_COL in out else 0
        ror = out[RUN_OF_RIVER_COL] if RUN_OF_RIVER_COL in out else 0
        renewable_supply = wind + solar + ror
        out["residual_load"] = (out[load_col] - renewable_supply).astype(np.float32)
        out["renewable_penetration"] = (
            _safe_div(wind + solar, out[load_col]).clip(0, 1).fillna(0).astype(np.float32)
        )
    return out


# ---------------------------------------------------------------- spike

def add_spike(df: pd.DataFrame) -> pd.DataFrame:
    """Top-decile residual-load and renewable-scarcity flags."""
    out = df.copy()
    if "residual_load" in out:
        rl = out["residual_load"]
        thresholds = rl.groupby(out.index.hour).transform(
            lambda s: s.expanding(min_periods=24 * 30).quantile(0.9)
        )
        out["is_high_residual_load"] = (rl >= thresholds).astype(np.float32).fillna(0)
    if "renewable_penetration" in out:
        out["is_renewable_scarcity"] = (out["renewable_penetration"] < 0.10).astype(np.float32)
    return out


# ---------------------------------------------------------------- regime

CRISIS_START = pd.Timestamp("2022-03-01", tz="UTC")
CRISIS_END = pd.Timestamp("2023-03-01", tz="UTC")


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """2022-2023 European energy-crisis flag."""
    out = df.copy()
    out["crisis_2022_flag"] = (
        ((out.index >= CRISIS_START) & (out.index < CRISIS_END))
        .astype(np.float32)
    )
    return out


# ---------------------------------------------------------------- weather

def weather_columns(cities: Iterable[str] | None = None) -> tuple[str, ...]:
    cities = tuple(cities) if cities is not None else CITY_NAMES
    return tuple(f"{c}_{v}" for c in cities for v in WEATHER_VARIABLES)


def add_weather_lags(
    df: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (0, 1, 3, 6, 24),
    cities: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Append lagged weather features at the origin timestamp."""
    out = df.copy()
    cols = weather_columns(cities)
    for col in cols:
        if col not in out.columns:
            continue
        for lag in lags:
            if lag == 0:
                continue
            out[f"{col}_lag{lag}"] = out[col].shift(lag).astype(np.float32)
    return out


def add_weather_aggregates(
    df: pd.DataFrame,
    *,
    cities: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Append cross-city mean (and max for wind/radiation) per variable."""
    out = df.copy()
    cities = list(cities) if cities is not None else list(CITY_NAMES)
    for var in WEATHER_VARIABLES:
        cols = [f"{c}_{var}" for c in cities if f"{c}_{var}" in out.columns]
        if cols:
            out[f"weather_mean_{var}"] = out[cols].mean(axis=1).astype(np.float32)
            if "wind" in var or "radiation" in var:
                out[f"weather_max_{var}"] = out[cols].max(axis=1).astype(np.float32)
    return out


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Default weather bundle: lags + aggregates."""
    out = add_weather_lags(df)
    out = add_weather_aggregates(out)
    return out


# ---------------------------------------------------------------- load quantiles (Module A output)

_LOAD_QUANTILES_PATH = (
    pathlib.Path(__file__).parents[2] / "data" / "module_a" / "load_quantiles.parquet"
)
_load_quantiles_cache: pd.DataFrame | None = None


def _load_quantile_df() -> pd.DataFrame | None:
    global _load_quantiles_cache
    if _load_quantiles_cache is not None:
        return _load_quantiles_cache
    if not _LOAD_QUANTILES_PATH.exists():
        return None
    _load_quantiles_cache = pd.read_parquet(_LOAD_QUANTILES_PATH)
    return _load_quantiles_cache


def add_load_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Join Module A load quantile forecasts to the price feature frame.

    Adds 72 columns: load_q{10,50,90}_h{1..24} indexed by forecast origin.
    No-op (with warning) if data/module_a/load_quantiles.parquet is missing -
    run ``python -m module_a.train`` first to generate it.
    """
    lq = _load_quantile_df()
    if lq is None:
        import warnings
        warnings.warn(
            "load_quantiles.parquet not found - run module_a.train first. "
            "Skipping load_quantiles bundle.",
            stacklevel=2,
        )
        return df
    out = df.copy()
    aligned = lq.reindex(out.index)
    for col in aligned.columns:
        out[col] = aligned[col].astype(np.float32)
    return out


# ============================================================ registry

BundleFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class _BundleSpec:
    """One feature bundle: a function plus its dependency bundle names."""

    fn: BundleFn
    requires: tuple[str, ...] = ()
    name: str = ""


class _Registry(dict):
    """``dict[name -> _BundleSpec]`` with a topological resolver."""

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
    "calendar":       _BundleSpec(add_calendar,      name="calendar"),
    "lags":           _BundleSpec(add_price_lags,    name="lags"),
    "fundamentals":   _BundleSpec(add_fundamentals,  name="fundamentals"),
    "spike":          _BundleSpec(add_spike,         requires=("fundamentals",), name="spike"),
    "regime":         _BundleSpec(add_regime,        name="regime"),
    "weather":        _BundleSpec(add_weather,       name="weather"),
    "load_quantiles": _BundleSpec(add_load_quantiles, name="load_quantiles"),
})


def build_features(df: pd.DataFrame, bundles: Sequence[str]) -> pd.DataFrame:
    """Apply the requested bundles in dependency order."""
    out = df
    for spec in REGISTRY.resolve_order(bundles):
        out = spec.fn(out)
    return out


# ============================================================ supervised layout

HORIZON_COL = "horizon_h"
ORIGIN_COL = "origin_ts"
TARGET_COL = "target_ts"


def prepare_supervised(
    feat_df: pd.DataFrame,
    *,
    horizons: Iterable[int] = range(1, HORIZON + 1),
    past_cols: Sequence[str],
    future_cols: Sequence[str],
    target_col: str = PRICE_COL,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the flat (origin, horizon) supervised-learning layout."""
    horizons = list(horizons)
    past_df = feat_df[list(past_cols)]
    future_df = feat_df[list(future_cols)] if future_cols else None

    chunks_X: list[pd.DataFrame] = []
    chunks_y: list[pd.Series] = []

    for h in horizons:
        target = feat_df[target_col].shift(-h)
        block = past_df.copy()
        if future_df is not None:
            shifted_future = future_df.shift(-h).add_prefix("fut_")
            block = pd.concat([block, shifted_future], axis=1)
        block[HORIZON_COL] = np.int16(h)
        block[ORIGIN_COL] = block.index
        block[TARGET_COL] = block.index + pd.Timedelta(hours=h)
        valid = target.notna()
        if drop_na:
            valid &= block.notna().all(axis=1)
        chunks_X.append(block.loc[valid])
        chunks_y.append(target.loc[valid])

    X = pd.concat(chunks_X, axis=0).reset_index(drop=True)
    y = pd.concat(chunks_y, axis=0).reset_index(drop=True)
    y.name = target_col
    return X, y


# ============================================================ horizon groups


class HorizonGroup(str, Enum):
    H1_6 = "h1_6"
    H7_18 = "h7_18"
    H19_24 = "h19_24"


HORIZON_RANGES: dict[HorizonGroup, range] = {
    HorizonGroup.H1_6: range(1, 7),
    HorizonGroup.H7_18: range(7, 19),
    HorizonGroup.H19_24: range(19, 25),
}


def filter_by_horizon(
    X: pd.DataFrame,
    y: pd.Series,
    horizons: tuple[int, ...] | range,
    horizon_col: str = HORIZON_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter (X, y) rows to those whose ``horizon_h`` is in ``horizons``."""
    if horizon_col not in X.columns:
        raise KeyError(f"X is missing the horizon column {horizon_col!r}")
    mask = X[horizon_col].isin(list(horizons))
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
