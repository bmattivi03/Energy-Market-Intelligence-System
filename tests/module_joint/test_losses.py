import torch

from module_joint.losses import (
    DWA,
    UncertaintyWeighting,
    horizon_weights,
    multi_quantile_pinball,
    pinball_q,
)


def test_pinball_matches_reference():
    pred = torch.zeros(4, 24)
    target = torch.ones(4, 24)
    # alpha=0.9, underprediction by 1 -> 0.9 * 1
    assert torch.allclose(pinball_q(pred, target, 0.9), torch.tensor(0.9), atol=1e-6)
    # alpha=0.1, underprediction by 1 -> 0.1 * 1
    assert torch.allclose(pinball_q(pred, target, 0.1), torch.tensor(0.1), atol=1e-6)


def test_multi_quantile_zero_for_perfect():
    target = torch.randn(4, 24)
    pred = target.unsqueeze(-1).repeat(1, 1, 3)
    assert multi_quantile_pinball(pred, target, (0.1, 0.5, 0.9)) < 1e-6


def test_horizon_weights_emphasize_segment():
    pred = torch.zeros(2, 24, 3)
    target = torch.ones(2, 24)
    w = horizon_weights(24, 2.0)
    assert w[0] == 2.0 and w[6] == 1.0
    hi = multi_quantile_pinball(pred, target, (0.1, 0.5, 0.9), horizon_weights=w)
    lo = multi_quantile_pinball(pred, target, (0.1, 0.5, 0.9))
    assert hi > lo


def test_dwa_warmup_equal_then_adapts():
    dwa = DWA(2)
    w0 = dwa.weights([1.0, 1.0])
    assert torch.allclose(w0, torch.ones(2))
    dwa.weights([0.9, 0.5])
    w2 = dwa.weights([0.8, 0.2])
    assert w2.shape == (2,)
    assert torch.allclose(w2.sum(), torch.tensor(2.0), atol=1e-5)


def test_uncertainty_weighting_differentiable():
    uw = UncertaintyWeighting(2)
    loss = uw([torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)])
    loss.backward()
    assert uw.log_var.grad is not None
