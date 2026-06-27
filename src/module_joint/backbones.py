"""Shared-encoder backbones. Each maps a past window (B, lookback, n_feat) to a
shared latent z of size .out_dim, behind one common interface so the joint model
and the single-task bench can swap them freely.

TiDE is the default (fast residual-MLP, MPS-friendly, strong on smooth load).
DLinear is the linear floor. NHiTSx is the multi-rate MLP fallback / ensemble
member.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class Backbone(Protocol):
    out_dim: int

    def forward(self, past: torch.Tensor) -> torch.Tensor: ...


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TiDEBackbone(nn.Module):
    """Dense encoder: flatten (lookback x features) then a residual-MLP stack."""

    def __init__(self, n_past_feat, lookback, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.out_dim = hidden
        self.inp = nn.Linear(n_past_feat * lookback, hidden)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden, dropout) for _ in range(layers)]
        )

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        x = self.inp(past.flatten(1))
        for b in self.blocks:
            x = b(x)
        return x


class DLinearBackbone(nn.Module):
    """Decomposition-linear encoder: trend (moving average) + seasonal, each a
    linear map of the flattened window, summed into the latent.
    """

    def __init__(self, n_past_feat, lookback, hidden=256, kernel=25, **_):
        super().__init__()
        self.out_dim = hidden
        self.kernel = kernel
        flat = n_past_feat * lookback
        self.trend = nn.Linear(flat, hidden)
        self.seasonal = nn.Linear(flat, hidden)

    def _moving_avg(self, x):
        # x: (B, L, F) -> trend via avg pool along time with reflection padding
        pad = self.kernel // 2
        xt = x.transpose(1, 2)  # (B, F, L)
        xt = torch.nn.functional.pad(xt, (pad, pad), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(xt, self.kernel, stride=1)
        return trend.transpose(1, 2)[:, : x.shape[1], :]

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        trend = self._moving_avg(past)
        seasonal = past - trend
        return self.trend(trend.flatten(1)) + self.seasonal(seasonal.flatten(1))


class NHiTSxBackbone(nn.Module):
    """Multi-rate hierarchical MLP: pool the window at several rates, encode each
    pooled view, and sum. A lightweight N-HiTS-style stack for the latent.
    """

    def __init__(self, n_past_feat, lookback, hidden=256, rates=(1, 2, 4), dropout=0.2):
        super().__init__()
        self.out_dim = hidden
        self.rates = rates
        self.encoders = nn.ModuleList()
        for r in rates:
            pooled_len = lookback // r
            self.encoders.append(
                nn.Sequential(
                    nn.Linear(n_past_feat * pooled_len, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
            )

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        z = 0.0
        for r, enc in zip(self.rates, self.encoders):
            if r == 1:
                view = past
            else:
                xt = past.transpose(1, 2)
                view = torch.nn.functional.avg_pool1d(xt, r, stride=r).transpose(1, 2)
            z = z + enc(view.flatten(1))
        return z


def build_backbone(name: str, n_past_feat: int, lookback: int, cfg) -> nn.Module:
    name = name.lower()
    if name == "tide":
        return TiDEBackbone(n_past_feat, lookback, cfg.hidden, cfg.enc_layers, cfg.dropout)
    if name == "dlinear":
        return DLinearBackbone(n_past_feat, lookback, cfg.hidden)
    if name == "nhitsx":
        return NHiTSxBackbone(n_past_feat, lookback, cfg.hidden, dropout=cfg.dropout)
    raise ValueError(f"unknown backbone: {name}")
