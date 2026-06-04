"""Merge 6 raw parquets into data/processed/emis_raw.parquet.

Output: hourly UTC DatetimeIndex, 47 columns (no net_import_* - those are
recomputed by impute.py after imputation for flow-consistency).

Column count:
    price (1) + load (1) + gen_* (17) + flow (6) + weather (20) + fuels (2) = 47
"""

from __future__ import annotations

import pathlib
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed" / "emis_raw.parquet"

FLOW_COLS = ["AT_to_DELU", "CH_to_DELU", "DELU_to_AT", "DELU_to_CH", "DELU_to_FR", "FR_to_DELU"]
CITY_MAP = {"Berlin": "berlin", "Cologne": "cologne", "Frankfurt": "frankfurt",
            "Hamburg": "hamburg", "Munich": "munich"}
WEATHER_VARS = ["temperature_2m", "wind_speed_10m", "wind_speed_100m", "shortwave_radiation"]


def _load_prices() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "entsoe_prices_2019_2025.parquet")[["value"]]
    return df.rename(columns={"value": "price"})


def _load_load() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "entsoe_load_2019_2025.parquet")[["value"]]
    return df.rename(columns={"value": "load"})


def _load_generation() -> pd.DataFrame:
    return pd.read_parquet(RAW / "entsoe_generation_2019_2025.parquet")


def _load_crossborder() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "entsoe_crossborder_2019_2025.parquet")
    return df[FLOW_COLS]


def _load_weather() -> pd.DataFrame:
    raw = pd.read_parquet(RAW / "weather_2019_2025.parquet")
    # MultiIndex (datetime_utc, location) → wide format
    frames = []
    for city_orig, city_lower in CITY_MAP.items():
        city_df = raw.xs(city_orig, level="location")[WEATHER_VARS]
        city_df = city_df.rename(columns={v: f"{city_lower}_{v}" for v in WEATHER_VARS})
        frames.append(city_df)
    return pd.concat(frames, axis=1)


def _load_fuels() -> pd.DataFrame:
    fuels = pd.read_parquet(RAW / "fuels_2019_2025.parquet")[["ttf_gas", "carbon_ets"]]
    # Daily → hourly: forward-fill within each day, then reindex to hourly grid below
    return fuels


def build_emis_raw() -> pd.DataFrame:
    print("Loading raw sources...")
    prices = _load_prices()
    load   = _load_load()
    gen    = _load_generation()
    flows  = _load_crossborder()
    weather = _load_weather()
    fuels  = _load_fuels()

    # Deduplicate all sources (keep last, matching ingestion convention)
    for df in (prices, load, gen, flows, weather, fuels):
        dup = df.index.duplicated(keep="last").sum()
        if dup:
            print(f"  Dropping {dup} duplicate index entries from {df.columns.tolist()[:2]}")
    prices  = prices[~prices.index.duplicated(keep="last")]
    load    = load[~load.index.duplicated(keep="last")]
    gen     = gen[~gen.index.duplicated(keep="last")]
    flows   = flows[~flows.index.duplicated(keep="last")]
    weather = weather[~weather.index.duplicated(keep="last")]
    fuels   = fuels[~fuels.index.duplicated(keep="last")]

    # Fixed hourly UTC index 2019-01-01 00:00 → 2025-03-31 23:00 (test set end)
    hourly_idx = pd.date_range("2019-01-01", "2025-12-31 23:00", freq="h", tz="UTC",
                               name="datetime_utc")
    print(f"Hourly index: {hourly_idx[0]} to {hourly_idx[-1]}  ({len(hourly_idx):,} rows)")

    # Reindex all hourly sources (NaN for missing hours, keep as-is for imputer)
    frames = [
        prices.reindex(hourly_idx),
        load.reindex(hourly_idx),
        gen.reindex(hourly_idx),
        flows.reindex(hourly_idx),
        weather.reindex(hourly_idx),
    ]

    # Fuels: daily → ffill to hourly grid
    fuels_hourly = fuels.reindex(hourly_idx).ffill()
    frames.append(fuels_hourly)

    merged = pd.concat(frames, axis=1)
    merged.index.name = "datetime_utc"

    print(f"Merged shape: {merged.shape}")
    nan_pct = merged.isnull().mean()
    noisy = nan_pct[nan_pct > 0.01]
    if not noisy.empty:
        print(f"Columns with >1% NaN:\n{noisy.round(3)}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT)
    print(f"Saved: {OUT}")
    return merged


if __name__ == "__main__":
    build_emis_raw()
