"""Multi-task losses: pinball (quantile) loss, dynamic weight averaging (DWA),
homoscedastic uncertainty weighting, and the combined joint objective.

Target normalization (asinh price + standardize) is done in data.py, so by the
time losses are computed the per-task pinball terms are on comparable scales and
fixed/DWA weighting is competitive with learned uncertainty weighting.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pinball_q(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean pinball loss at a single quantile level. pred/target: (B, H)."""
    diff = target - pred
    return torch.mean(torch.maximum(alpha * diff, (alpha - 1) * diff))


def multi_quantile_pinball(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average pinball over quantiles. pred: (B, H, Q), target: (B, H).

    horizon_weights: optional (H,) tensor to emphasize specific horizons
    (the headline h1-6 segment).
    """
    losses = []
    for i, a in enumerate(quantiles):
        diff = target - pred[..., i]
        pl = torch.maximum(a * diff, (a - 1) * diff)  # (B, H)
        if horizon_weights is not None:
            pl = pl * horizon_weights.to(pl.device)
        losses.append(pl.mean())
    return torch.stack(losses).mean()


class DWA:
    """Dynamic Weight Averaging (Liu et al. 2019).

    Tasks whose loss is decreasing slowly get up-weighted. Needs two prior
    epochs of per-task losses to warm up; returns equal weights before that.
    """

    def __init__(self, n_tasks: int, temp: float = 2.0):
        self.n = n_tasks
        self.temp = temp
        self.prev = None
        self.prev2 = None

    def weights(self, losses) -> torch.Tensor:
        vals = [float(x.detach()) if torch.is_tensor(x) else float(x) for x in losses]
        if self.prev is None or self.prev2 is None:
            w = torch.ones(self.n)
        else:
            r = torch.tensor(
                [self.prev[i] / (self.prev2[i] + 1e-8) for i in range(self.n)]
            )
            w = self.n * torch.softmax(r / self.temp, dim=0)
        self.prev2 = self.prev
        self.prev = vals
        return w


class UncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall, Gal, Cipolla 2018)."""

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses) -> torch.Tensor:
        total = 0.0
        for i, l in enumerate(losses):
            total = total + torch.exp(-self.log_var[i]) * l + 0.5 * self.log_var[i]
        return total


def horizon_weights(horizon: int, h1_6_weight: float) -> torch.Tensor:
    """(H,) weights with the first 6 horizons up-weighted."""
    w = torch.ones(horizon)
    w[:6] = h1_6_weight
    return w


def joint_loss(outputs: dict, targets: dict, cfg, weighter) -> tuple:
    """Combine the two primary pinball terms, the aux regressions, and the
    spike BCE into a single scalar using the chosen weighter.

    outputs: dict with load/price (B,H,3), residual_load/renewables (B,H,*),
             spike (B,H) logits.
    targets: dict with load/price (B,H), residual_load/renewables (B,H,*),
             spike (B,H) in {0,1}.
    Returns (total_loss, parts_dict).
    """
    quantiles = (0.1, 0.5, 0.9)
    hw = horizon_weights(cfg.horizon, cfg.horizon_weight_h1_6).to(
        outputs["load"].device
    )
    parts = {}
    parts["load"] = multi_quantile_pinball(outputs["load"], targets["load"], quantiles, hw)
    parts["price"] = multi_quantile_pinball(outputs["price"], targets["price"], quantiles, hw)

    primary = [parts["load"], parts["price"]]
    if cfg.loss_weighting == "uncertainty":
        primary_total = weighter(primary)
    elif cfg.loss_weighting == "fixed":
        primary_total = (
            cfg.task_weights["load"] * parts["load"]
            + cfg.task_weights["price"] * parts["price"]
        )
    else:  # dwa
        w = weighter.weights(primary).to(outputs["load"].device)
        primary_total = w[0] * parts["load"] + w[1] * parts["price"]

    aux_total = 0.0
    if cfg.aux_residual_load and "residual_load" in outputs:
        parts["residual_load"] = F.mse_loss(
            outputs["residual_load"].squeeze(-1), targets["residual_load"]
        )
        aux_total = aux_total + parts["residual_load"]
    if cfg.aux_renewables and "renewables" in outputs:
        parts["renewables"] = F.mse_loss(outputs["renewables"], targets["renewables"])
        aux_total = aux_total + parts["renewables"]

    spike_total = 0.0
    if cfg.aux_spike and "spike" in outputs:
        parts["spike"] = F.binary_cross_entropy_with_logits(
            outputs["spike"], targets["spike"]
        )
        spike_total = parts["spike"]

    total = primary_total + cfg.aux_weight * aux_total + cfg.spike_weight * spike_total
    parts["total"] = total
    return total, parts
