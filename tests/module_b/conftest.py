import numpy as np
import pandas as pd
import pytest

FLOAT_COLS = [
    "price", "load", "gen_biomass", "gen_fossil_lignite", "gen_fossil_coal_gas",
    "gen_fossil_gas", "gen_fossil_hard_coal", "gen_fossil_oil", "gen_geothermal",
    "gen_hydro_pumped", "gen_hydro_ror", "gen_hydro_reservoir", "gen_nuclear",
    "gen_other_renewable", "gen_solar", "gen_waste", "gen_wind_offshore",
    "gen_wind_onshore", "gen_other",
    "AT_to_DELU", "CH_to_DELU", "DELU_to_AT", "DELU_to_CH", "DELU_to_FR", "FR_to_DELU",
    "berlin_temperature_2m", "berlin_wind_speed_10m", "berlin_wind_speed_100m",
    "berlin_shortwave_radiation", "cologne_temperature_2m", "cologne_wind_speed_10m",
    "cologne_wind_speed_100m", "cologne_shortwave_radiation",
    "frankfurt_temperature_2m", "frankfurt_wind_speed_10m", "frankfurt_wind_speed_100m",
    "frankfurt_shortwave_radiation", "hamburg_temperature_2m", "hamburg_wind_speed_10m",
    "hamburg_wind_speed_100m", "hamburg_shortwave_radiation",
    "munich_temperature_2m", "munich_wind_speed_10m", "munich_wind_speed_100m",
    "munich_shortwave_radiation",
    "ttf_gas", "carbon_ets", "net_import_FR", "net_import_AT", "net_import_CH",
]


@pytest.fixture
def small_df():
    """400h starting 2023-01-01 with all 51 split columns."""
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC", name="datetime_utc")
    rng = np.random.default_rng(42)
    data = {col: np.abs(rng.random(n)) * 100 + 1.0 for col in FLOAT_COLS}
    data["gen_fossil_coal_gas_structural_zero"] = np.zeros(n)
    return pd.DataFrame(data, index=idx)
