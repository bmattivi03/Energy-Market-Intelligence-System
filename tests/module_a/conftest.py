"""Synthetic fixtures for Module A (load forecasting) tests.

Mirrors the ``small_df`` pattern in ``tests/module_b/conftest.py`` but tailored
to the columns ``module_a.features.build_features(ALL_BUNDLES)`` actually
consumes:

* ``load`` - the target (``TARGET_COL``) and the load-lag autoregressive signal.
* ``{city}_{var}`` weather columns for the 5 German cities × 4 variables -
  consumed by ``add_weather``.
* ``gen_wind_onshore``, ``gen_wind_offshore``, ``gen_solar`` - consumed by
  ``add_renewables``.

The DataFrame has a deterministic (seeded), tz-aware UTC hourly index. It is
sized so that, after ``build_features`` (max lag 336h) + the LSTM 168h
lookback + 24h horizon, the ``LoadSequenceDataset`` still yields several valid
training origins.
"""

import numpy as np
import pandas as pd
import pytest

from data.schemas import CITY_NAMES, WEATHER_VARIABLES

# Columns build_features() reads (besides the calendar index).
WEATHER_COLS = [f"{c}_{v}" for c in CITY_NAMES for v in WEATHER_VARIABLES]
RENEWABLE_GEN_COLS = ["gen_wind_onshore", "gen_wind_offshore", "gen_solar"]

N_ROWS = 520  # > 336 (max lag) + 168 (lookback) + 24 (horizon) -> usable origins


def _make_load_df(n: int = N_ROWS, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic load + weather + renewables frame."""
    idx = pd.date_range(
        "2023-01-01", periods=n, freq="h", tz="UTC", name="datetime_utc"
    )
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Load: daily + weekly seasonality + noise, kept strictly positive (MW scale).
    load = (
        30000.0
        + 8000.0 * np.sin(2 * np.pi * t / 24)
        + 3000.0 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 800, n)
    )

    data = {"load": load.astype(np.float64)}

    # Weather: temperature with a daily cycle, wind/radiation as positive noise.
    for col in WEATHER_COLS:
        if col.endswith("temperature_2m"):
            data[col] = (
                10.0 + 6.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1.5, n)
            ).astype(np.float64)
        elif col.endswith("shortwave_radiation"):
            # Daylight-only radiation: zero at night, positive midday.
            rad = np.clip(np.sin(2 * np.pi * (t % 24) / 24), 0, None) * 600.0
            data[col] = (rad + rng.random(n) * 20.0).astype(np.float64)
        else:  # wind speeds
            data[col] = (np.abs(rng.normal(5, 2, n))).astype(np.float64)

    # Renewable generation totals (MW).
    for col in RENEWABLE_GEN_COLS:
        data[col] = (np.abs(rng.normal(4000, 1500, n))).astype(np.float64)

    return pd.DataFrame(data, index=idx)


@pytest.fixture
def load_df() -> pd.DataFrame:
    """520h of deterministic load/weather/renewables data on a UTC index."""
    return _make_load_df()


@pytest.fixture
def split_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A (train, val) pair carved from one contiguous deterministic series.

    Both share the same generative process so a scaler fit on ``train`` is a
    sensible transform for ``val`` - but the splits never overlap in time.
    """
    full = _make_load_df(n=N_ROWS + 240)
    train = full.iloc[:N_ROWS]
    val = full.iloc[N_ROWS:]
    return train, val
