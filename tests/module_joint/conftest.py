"""Shared fixtures for module_joint tests."""
import numpy as np
import pandas as pd
import pytest

from data import schemas


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end tests that train tiny models")


def _all_base_cols():
    return (
        [schemas.PRICE_COL, schemas.LOAD_COL]
        + list(schemas.GENERATION_COLS)
        + list(schemas.WEATHER_COLS)
        + list(schemas.CROSS_BORDER_FLOW_COLS)
        + list(schemas.NET_IMPORT_COLS)
        + list(schemas.FUEL_COLS)
    )


@pytest.fixture
def synthetic_df():
    """400h hourly UTC frame with all base schema columns, deterministic.

    Load and price carry a clean daily cycle so models can actually learn
    something in the tiny smoke trainings.
    """
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    t = np.arange(n)
    daily = np.sin(2 * np.pi * t / 24)
    data = {c: rng.normal(0, 1, n) + daily for c in _all_base_cols()}
    df = pd.DataFrame(data, index=idx)
    df[schemas.LOAD_COL] = 50000 + 5000 * daily + rng.normal(0, 200, n)
    df[schemas.PRICE_COL] = 80 + 30 * daily + rng.normal(0, 5, n)
    # a couple of renewable generation columns referenced by residual-load aux
    for c in ("gen_wind_onshore", "gen_wind_offshore", "gen_solar", "gen_hydro_ror"):
        if c in df.columns:
            df[c] = np.abs(rng.normal(5000, 1000, n))
    return df


@pytest.fixture
def fake_load_quantiles(synthetic_df):
    """Module-A-style export aligned to synthetic_df origins."""
    cols = [f"load_q{q}_h{h}" for q in (10, 50, 90) for h in range(1, 25)]
    rng = np.random.default_rng(1)
    base = rng.normal(50000, 1000, (len(synthetic_df), len(cols)))
    return pd.DataFrame(base, index=synthetic_df.index, columns=cols)
