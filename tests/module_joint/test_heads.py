import torch

from module_joint.heads import MonotoneQuantileHead, RegressionHead, SpikeHead


def test_quantile_head_shape_and_monotone():
    head = MonotoneQuantileHead(in_dim=16, horizon=24)
    out = head(torch.randn(8, 16))
    assert out.shape == (8, 24, 3)
    assert torch.all(out[..., 0] <= out[..., 1] + 1e-6)
    assert torch.all(out[..., 1] <= out[..., 2] + 1e-6)


def test_quantile_head_batch_agnostic():
    head = MonotoneQuantileHead(in_dim=16, horizon=24)
    assert head(torch.randn(1, 16)).shape == (1, 24, 3)
    assert head(torch.randn(5, 16)).shape == (5, 24, 3)


def test_regression_head_shape():
    head = RegressionHead(in_dim=16, horizon=24, out=2)
    assert head(torch.randn(8, 16)).shape == (8, 24, 2)


def test_spike_head_shape():
    head = SpikeHead(in_dim=16, horizon=24)
    assert head(torch.randn(8, 16)).shape == (8, 24)
