"""Tests for module_a.model — MultiScaleLSTM + forecaster wrapper.

Uses a TINY model (small hidden dim, very few epochs) on synthetic data, on
CPU, so the whole module runs in a few seconds. ``_select_device`` is patched
to CPU (it ignores EMIS_FORCE_CPU and probes the hardware directly).
"""

import numpy as np
import pandas as pd
import pytest
import torch

import module_a.model as m
from module_a.model import (
    HORIZON,
    LONG_WINDOW,
    QUANTILES,
    SHORT_WINDOW,
    LoadSequenceDataset,
    MultiScaleLSTM,
    MultiScaleLSTMForecaster,
    pinball_loss,
)
from module_a.features import ALL_BUNDLES, TARGET_COL, build_features


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Pin every device selection to CPU for speed + determinism."""
    cpu = torch.device("cpu")
    monkeypatch.setattr(m, "_select_device", lambda: cpu)
    torch.manual_seed(0)


@pytest.fixture
def feat_df(load_df):
    return build_features(load_df, ALL_BUNDLES)


def _tiny_forecaster(**kw):
    defaults = dict(
        hidden=8,
        dropout=0.0,
        num_layers_short=1,
        num_layers_long=1,
        batch_size=16,
        max_epochs=2,
        patience=5,
        random_state=42,
    )
    defaults.update(kw)
    return MultiScaleLSTMForecaster(**defaults)


# ---------------------------------------------------------------- constants


def test_module_constants():
    assert QUANTILES == (0.1, 0.5, 0.9)
    assert HORIZON == 24
    assert LONG_WINDOW == 168
    assert SHORT_WINDOW == 48


# ---------------------------------------------------------------- raw nn.Module


def test_forward_output_shape():
    n_features = 11
    model = MultiScaleLSTM(n_features, hidden=8, dropout=0.0,
                           num_layers_short=1, num_layers_long=1)
    B = 4
    x_short = torch.randn(B, SHORT_WINDOW, n_features)
    x_long = torch.randn(B, LONG_WINDOW, n_features)
    out = model(x_short, x_long)
    # (B, H, Q) == (4, 24, 3)
    assert out.shape == (B, HORIZON, len(QUANTILES))


# ---------------------------------------------------------------- dataset


def test_dataset_item_shapes(feat_df):
    fc = MultiScaleLSTMForecaster()
    fc._feat_cols = fc._feature_cols(feat_df)
    X, y = fc._to_arrays(feat_df, fit_scalers=True)
    ds = LoadSequenceDataset(X, y, stride=1)
    assert len(ds) > 0
    x_short, x_long, target = ds[0]
    assert x_short.shape == (SHORT_WINDOW, X.shape[1])
    assert x_long.shape == (LONG_WINDOW, X.shape[1])
    assert target.shape == (HORIZON,)
    assert x_short.dtype == torch.float32


def test_dataset_origin_alignment(feat_df):
    """x_long must be the LOOKBACK rows immediately before the target window."""
    fc = MultiScaleLSTMForecaster()
    fc._feat_cols = fc._feature_cols(feat_df)
    X, y = fc._to_arrays(feat_df, fit_scalers=True)
    ds = LoadSequenceDataset(X, y, stride=1)
    t = ds.origins[0]
    x_short, x_long, target = ds[0]
    np.testing.assert_array_equal(x_long.numpy(), X[t - LONG_WINDOW:t])
    np.testing.assert_array_equal(x_short.numpy(), X[t - SHORT_WINDOW:t])
    np.testing.assert_array_equal(target.numpy(), y[t:t + HORIZON])


# ---------------------------------------------------------------- pinball loss


def test_pinball_loss_finite_and_non_negative():
    B = 5
    pred = torch.randn(B, HORIZON, len(QUANTILES))
    target = torch.randn(B, HORIZON)
    loss = pinball_loss(pred, target, QUANTILES)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_pinball_loss_zero_for_perfect_median_prediction():
    """If all quantiles equal the target, only asymmetry of q remains.

    With pred==target everywhere, err==0 so the loss is exactly 0.
    """
    B = 3
    target = torch.randn(B, HORIZON)
    pred = target.unsqueeze(-1).repeat(1, 1, len(QUANTILES))
    loss = pinball_loss(pred, target, QUANTILES)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pinball_loss_matches_numpy_reference():
    torch.manual_seed(3)
    B = 4
    pred = torch.randn(B, HORIZON, len(QUANTILES))
    target = torch.randn(B, HORIZON)
    loss = pinball_loss(pred, target, QUANTILES).item()

    q = np.array(QUANTILES)
    p = pred.numpy()
    t = target.numpy()[..., None]
    err = t - p
    ref = np.mean(np.where(err >= 0, q * err, (q - 1) * err))
    assert loss == pytest.approx(ref, rel=1e-5)


# ---------------------------------------------------------------- scaler round-trip


def test_scaler_round_trips(feat_df):
    """Target inverse-scale recovers the original load values."""
    fc = _tiny_forecaster()
    fc._feat_cols = fc._feature_cols(feat_df)
    X_sc, y_sc = fc._to_arrays(feat_df, fit_scalers=True)

    # Inverse the standardisation the way predict_quantiles does.
    mean = fc._tgt_scaler.mean_[0]
    std = fc._tgt_scaler.scale_[0]
    y_recovered = y_sc * std + mean

    y_orig = feat_df[TARGET_COL].to_numpy(np.float64)
    np.testing.assert_allclose(y_recovered, y_orig, rtol=1e-4, atol=1e-2)

    # The feature scaler round-trips: inverse_transform(transform(X)) == X on
    # the rows with no NaN (the late rows past the 336h lag warm-up).
    X_raw = feat_df[fc._feat_cols].to_numpy(np.float64)
    finite_rows = np.isfinite(X_raw).all(axis=1)
    assert finite_rows.sum() > 0
    Xf = X_raw[finite_rows]
    Z = fc._feat_scaler.transform(Xf)
    X_back = fc._feat_scaler.inverse_transform(Z)
    np.testing.assert_allclose(X_back, Xf, rtol=1e-5, atol=1e-4)


# ---------------------------------------------------------------- fit / predict


def test_fit_predict_output_shape_and_columns(feat_df):
    fc = _tiny_forecaster()
    fc.fit(feat_df)
    preds = fc.predict_quantiles(feat_df)

    assert list(preds.columns) == ["q10", "q50", "q90"]
    assert preds.index.names == ["origin_ts", "horizon_h"]

    # One row per (origin, horizon). #origins = len(range(168, T-24)).
    T = len(feat_df)
    n_origins = len(range(LONG_WINDOW, T - HORIZON, 1))
    assert len(preds) == n_origins * HORIZON

    horizons = preds.index.get_level_values("horizon_h").unique().tolist()
    assert sorted(horizons) == list(range(1, HORIZON + 1))
    assert np.isfinite(preds.to_numpy()).all()


def test_predict_before_fit_raises(feat_df):
    fc = _tiny_forecaster()
    with pytest.raises(RuntimeError):
        fc.predict_quantiles(feat_df)


# ---------------------------------------------------------------- checkpoint round-trip


def test_checkpoint_save_load_reproduces_predictions(feat_df, tmp_path):
    fc = _tiny_forecaster()
    fc.fit(feat_df)
    preds_before = fc.predict_quantiles(feat_df)

    ckpt = tmp_path / "module_a_tiny.pt"
    fc.save(ckpt)
    assert ckpt.exists()

    reloaded = MultiScaleLSTMForecaster.load(ckpt)

    # Reloaded hyperparameters and feature columns match.
    assert reloaded._feat_cols == fc._feat_cols
    assert reloaded.hidden == fc.hidden

    preds_after = reloaded.predict_quantiles(feat_df)

    pd.testing.assert_frame_equal(
        preds_before, preds_after, rtol=1e-5, atol=1e-4
    )
