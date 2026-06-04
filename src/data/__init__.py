"""Global data layer for the Energy Market Intelligence System.

Owns: loading the splits parquets, schema constants, mask-aware accessors,
rolling-origin CV iterators. All ML modules (Module A/B/C) consume data
through this layer; nothing in the ML modules reads parquets directly.
"""

from .loaders import load_split, load_imputed, load_raw, load_mask
from .schemas import (
    PRICE_COL,
    LOAD_COL,
    GENERATION_COLS,
    WEATHER_COLS,
    CROSS_BORDER_FLOW_COLS,
    NET_IMPORT_COLS,
    FUEL_COLS,
    CITY_NAMES,
    CITY_COORDS,
    SPLIT_BOUNDS,
)
from .splits import (
    rolling_origin_folds,
    expanding_window_folds,
    HORIZON,
    LOOKBACK,
    Fold,
)

__all__ = [
    "load_split", "load_imputed", "load_raw", "load_mask",
    "PRICE_COL", "LOAD_COL", "GENERATION_COLS", "WEATHER_COLS",
    "CROSS_BORDER_FLOW_COLS", "NET_IMPORT_COLS", "FUEL_COLS",
    "CITY_NAMES", "CITY_COORDS", "SPLIT_BOUNDS",
    "rolling_origin_folds", "expanding_window_folds", "HORIZON", "LOOKBACK", "Fold",
]
