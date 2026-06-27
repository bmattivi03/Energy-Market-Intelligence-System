"""Task-specific output heads off the shared latent / decoder features.

MonotoneQuantileHead emits non-crossing q10/q50/q90. RegressionHead is used for
the residual-load and renewables auxiliary targets. SpikeHead emits per-horizon
logits for the price-spike auxiliary classification task.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .transforms import monotone_quantiles


class MonotoneQuantileHead(nn.Module):
    """(B, in_dim) -> (B, horizon, 3) with q10 <= q50 <= q90 by construction."""

    def __init__(self, in_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.proj = nn.Linear(in_dim, horizon * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.proj(x).view(x.shape[0], self.horizon, 3)
        return monotone_quantiles(raw)


class RegressionHead(nn.Module):
    """(B, in_dim) -> (B, horizon, out)."""

    def __init__(self, in_dim: int, horizon: int, out: int = 1):
        super().__init__()
        self.horizon = horizon
        self.out = out
        self.proj = nn.Linear(in_dim, horizon * out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).view(x.shape[0], self.horizon, self.out)


class SpikeHead(nn.Module):
    """(B, in_dim) -> (B, horizon) logits for per-hour spike classification."""

    def __init__(self, in_dim: int, horizon: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
