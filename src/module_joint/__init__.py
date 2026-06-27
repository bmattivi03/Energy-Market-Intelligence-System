"""Joint multi-task deep learning forecaster for DE-LU load and price.

A single shared-encoder model that forecasts both system load (MW) and
day-ahead price (EUR/MWh) jointly as q10/q50/q90 over h1-24. Imports and
reuses src/data and src/module_b/evaluation; never modifies Module A or B.
"""
from .config import (
    ALPHA,
    HORIZON,
    LOOKBACK,
    QUANTILE_COLS,
    QUANTILES,
    JointConfig,
    select_device,
    set_seed,
)

__all__ = [
    "JointConfig",
    "HORIZON",
    "LOOKBACK",
    "QUANTILES",
    "QUANTILE_COLS",
    "ALPHA",
    "set_seed",
    "select_device",
]
