"""Module B — day-ahead electricity price forecasting.

Flat 3-file layout:

* :mod:`module_b.models` — base class, baselines, classical wrappers.
* :mod:`module_b.features` — feature transforms, REGISTRY, the supervised
  (origin, horizon) layout, and horizon-group helpers.
* :mod:`module_b.evaluation` — point/probabilistic metrics, segmentation,
  Diebold-Mariano, bootstrap, and conformal calibration wrappers.

Notebooks in ``notebooks/module_b/`` (B1–B3, B5–B6) drive every experiment.
This package only provides reusable building blocks.
"""

from module_b.evaluation import (
    AdaptiveConformalCalibrator,
    ConformalQuantileRegressor,
    SEGMENT_FNS,
    bootstrap_ci,
    coverage,
    diebold_mariano,
    directional_accuracy,
    mae,
    multi_pinball_loss,
    pinball_loss,
    rmse,
    segment_metrics,
    smape,
    spike_mae,
    winkler_score,
)
from module_b.features import (
    CRISIS_END,
    CRISIS_START,
    HORIZON_COL,
    HORIZON_RANGES,
    HorizonGroup,
    ORIGIN_COL,
    REGISTRY,
    TARGET_COL,
    add_calendar,
    add_fundamentals,
    add_price_lags,
    add_regime,
    add_spike,
    add_weather,
    build_features,
    filter_by_horizon,
    prepare_supervised,
)
from module_b.models import (
    BaseQuantileForecaster,
    CatBoostQuantileForecaster,
    LightGBMQuantileForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    quantile_columns,
)

__all__ = [
    # base + models
    "BaseQuantileForecaster", "quantile_columns",
    "NaiveForecaster", "SeasonalNaiveForecaster",
    "CatBoostQuantileForecaster", "LightGBMQuantileForecaster",
    # features + datasets + horizon
    "REGISTRY", "build_features",
    "add_calendar", "add_price_lags", "add_fundamentals",
    "add_spike", "add_regime", "add_weather",
    "CRISIS_START", "CRISIS_END",
    "HORIZON_COL", "ORIGIN_COL", "TARGET_COL", "prepare_supervised",
    "HorizonGroup", "HORIZON_RANGES", "filter_by_horizon",
    # evaluation
    "mae", "rmse", "smape", "pinball_loss", "multi_pinball_loss",
    "coverage", "winkler_score", "directional_accuracy", "spike_mae",
    "segment_metrics", "SEGMENT_FNS", "diebold_mariano", "bootstrap_ci",
    "ConformalQuantileRegressor", "AdaptiveConformalCalibrator",
]
