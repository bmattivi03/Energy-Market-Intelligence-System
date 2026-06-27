import numpy as np
import pytest
import torch

from module_joint.transforms import (
    FeatureScaler,
    RevIN,
    ScalarScaler,
    asinh,
    inv_asinh,
    monotone_quantiles,
)


def test_asinh_roundtrip_handles_negatives():
    x = np.array([-50.0, 0.0, 80.0, 3000.0])
    assert np.allclose(inv_asinh(asinh(x)), x, atol=1e-6)


def test_feature_scaler_train_only_and_inverse():
    a = np.array([[1.0, 10.0], [3.0, 30.0], [np.nan, 50.0]])
    s = FeatureScaler().fit(a)
    z = s.transform(a)
    assert np.nanmean(z[:, 0]) == pytest.approx(0.0, abs=1e-6)
    back = s.inverse(z)
    mask = ~np.isnan(a)
    assert np.allclose(back[mask], a[mask], atol=1e-6)


def test_feature_scaler_constant_column_no_nan():
    a = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    z = FeatureScaler().fit(a).transform(a)
    assert np.all(np.isfinite(z))
    assert np.allclose(z[:, 0], 0.0)


def test_scalar_scaler_roundtrip():
    a = np.array([10.0, 20.0, 30.0])
    s = ScalarScaler().fit(a)
    assert np.allclose(s.inverse(s.transform(a)), a, atol=1e-6)


def test_monotone_quantiles_never_cross():
    raw = torch.randn(64, 24, 3) * 5
    q = monotone_quantiles(raw)
    assert torch.all(q[..., 0] <= q[..., 1] + 1e-6)
    assert torch.all(q[..., 1] <= q[..., 2] + 1e-6)


def test_monotone_quantiles_median_is_free_base():
    raw = torch.zeros(2, 3, 3)
    raw[..., 0] = 7.0
    q = monotone_quantiles(raw)
    assert torch.allclose(q[..., 1], torch.full_like(q[..., 1], 7.0))


def test_revin_roundtrip():
    r = RevIN(4)
    x = torch.randn(8, 168, 4)
    norm = r(x, "norm")
    back = r(norm, "denorm")
    assert torch.allclose(back, x, atol=1e-4)
