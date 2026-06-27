"""Module A — DE-LU grid load forecasting.

Produces q10/q50/q90 quantile forecasts for 24-hour-ahead load.
These feed into Module B as input features.

Public API (once implemented):
    from module_a.features import build_features, prepare_supervised
    from module_a.model import MultiScaleLSTM
    from module_a.train import train, predict_quantiles
"""
