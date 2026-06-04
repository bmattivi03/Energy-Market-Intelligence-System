"""Fixtures for Module C (RL battery dispatch) tests.

All data is synthetic, seeded, and self-contained - no network, no parquet,
no checkpoints. The shapes mirror the real Module B → Module C contract:

* ``price`` - actual day-ahead clearing price (EUR/MWh).
* ``q10_h1`` .. ``q90_h24`` - 72 Module B quantile forecast columns
  (3 quantile labels × 24 horizons).
* optionally ``load_q50_h{1,6,12,24}`` - 4 load-quantile extras for the A→C path.
"""

import numpy as np
import pandas as pd
import pytest

from module_c.environment import _HORIZONS, _QUANTILE_LABELS

# 4 load-quantile extra columns exercising the A→C observation path.
LOAD_EXTRA_COLS = ("load_q50_h1", "load_q50_h6", "load_q50_h12", "load_q50_h24")


def _forecast_columns() -> list[str]:
    """The 72 Module B quantile forecast column names, in env order."""
    return [
        f"{q}_h{h}"
        for q in _QUANTILE_LABELS
        for h in range(1, _HORIZONS + 1)
    ]


def _make_price_df(n: int = 240, *, seed: int = 7, extras=()) -> pd.DataFrame:
    """Build a tz-aware UTC hourly price/forecast DataFrame.

    The price is a smooth diurnal sinusoid plus seeded noise so that there is
    real arbitrage structure (cheap nights, expensive evenings) for a policy to
    exploit. Forecast quantiles bracket the price (q10 < q50 < q90).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC", name="datetime_utc")

    hours = idx.hour.to_numpy()
    base = 50.0 + 30.0 * np.sin(2 * np.pi * (hours - 18) / 24.0)
    price = base + rng.normal(0.0, 5.0, size=n)

    data = {"price": price.astype(float)}

    # Quantile forecasts: centred on price, fanned out by horizon.
    for h in range(1, _HORIZONS + 1):
        spread = 3.0 + 0.2 * h
        centre = price + rng.normal(0.0, 2.0, size=n)
        data[f"q10_h{h}"] = (centre - spread).astype(float)
        data[f"q50_h{h}"] = centre.astype(float)
        data[f"q90_h{h}"] = (centre + spread).astype(float)

    for col in extras:
        # Plausible load-quantile magnitudes (MW), positive and varying.
        data[col] = (40_000.0 + rng.normal(0.0, 3_000.0, size=n)).astype(float)

    return pd.DataFrame(data, index=idx)


@pytest.fixture
def price_df() -> pd.DataFrame:
    """~240-row tz-aware hourly DataFrame: ``price`` + 72 forecast columns."""
    return _make_price_df()


@pytest.fixture
def price_df_with_load() -> pd.DataFrame:
    """``price_df`` plus the 4 load-quantile extra columns (A→C path)."""
    return _make_price_df(extras=LOAD_EXTRA_COLS)
