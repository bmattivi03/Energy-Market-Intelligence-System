"""Canonical column lists, dtypes, and constants for the EMIS dataset.

Single source of truth — every other module imports from here so that adding
or renaming a column requires changing exactly one file.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

PRICE_COL: Final = "price"
LOAD_COL: Final  = "load"

GENERATION_COLS: Final = (
    "gen_biomass", "gen_fossil_lignite", "gen_fossil_coal_gas", "gen_fossil_gas",
    "gen_fossil_hard_coal", "gen_fossil_oil", "gen_geothermal", "gen_hydro_pumped",
    "gen_hydro_ror", "gen_hydro_reservoir", "gen_nuclear", "gen_other_renewable",
    "gen_solar", "gen_waste", "gen_wind_offshore", "gen_wind_onshore", "gen_other",
)
RENEWABLE_COLS: Final = ("gen_wind_onshore", "gen_wind_offshore", "gen_solar", "gen_hydro_ror")

CITY_NAMES: Final = ("berlin", "cologne", "frankfurt", "hamburg", "munich")
CITY_COORDS: Final = {
    "berlin": (52.52, 13.41),
    "cologne": (50.94, 6.96),
    "frankfurt": (50.11, 8.68),
    "hamburg": (53.55, 9.99),
    "munich": (48.14, 11.58),
}
WEATHER_VARIABLES: Final = (
    "temperature_2m", "wind_speed_10m", "wind_speed_100m", "shortwave_radiation",
)
WEATHER_COLS: Final = tuple(
    f"{city}_{var}" for city in CITY_NAMES for var in WEATHER_VARIABLES
)

CROSS_BORDER_FLOW_COLS: Final = (
    "AT_to_DELU", "CH_to_DELU", "DELU_to_AT", "DELU_to_CH", "DELU_to_FR", "FR_to_DELU",
)
NET_IMPORT_COLS: Final = ("net_import_FR", "net_import_AT", "net_import_CH")

FUEL_COLS: Final = ("ttf_gas", "carbon_ets")

STRUCTURAL_ZERO_INDICATOR: Final = "gen_fossil_coal_gas_structural_zero"

# Train / val / test boundaries (UTC, hourly resolution).
SPLIT_BOUNDS: Final = {
    "train": (pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2023-12-31 23:00", tz="UTC")),
    "val":   (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:00", tz="UTC")),
    "test":  (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-03-31 23:00", tz="UTC")),
}

# Convenience aliases (used by module_a and impute.py).
TRAIN_START, TRAIN_END = SPLIT_BOUNDS["train"]
VAL_START,   VAL_END   = SPLIT_BOUNDS["val"]
TEST_START,  TEST_END  = SPLIT_BOUNDS["test"]
