"""Transforms: asinh for prices, train-only feature scaling, monotone quantile
parameterization, and RevIN instance normalization.

asinh is used instead of log because DE-LU day-ahead prices go negative.
The monotone quantile parameterization guarantees q10 <= q50 <= q90 by
construction, which is mandatory: crossing corrupts Winkler/coverage and
poisons the downstream conformal step.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def asinh(x):
    return np.arcsinh(np.asarray(x, dtype=float))


def inv_asinh(x):
    return np.sinh(np.asarray(x, dtype=float))


class FeatureScaler:
    """Per-column standardization fit on train rows only, NaN-safe.

    Constant columns get std=1 so they map to zero rather than NaN/inf.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, arr: np.ndarray) -> "FeatureScaler":
        a = np.asarray(arr, dtype=float)
        self.mean_ = np.nanmean(a, axis=0)
        std = np.nanstd(a, axis=0)
        std = np.where(std == 0, 1.0, std)
        self.std_ = std
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (np.asarray(arr, dtype=float) - self.mean_) / self.std_

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float) * self.std_ + self.mean_


class ScalarScaler:
    """Single-value standardization (for a 1-D target like load or price)."""

    def __init__(self):
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, arr: np.ndarray) -> "ScalarScaler":
        a = np.asarray(arr, dtype=float)
        self.mean_ = float(np.nanmean(a))
        s = float(np.nanstd(a))
        self.std_ = s if s != 0 else 1.0
        return self

    def transform(self, arr):
        return (np.asarray(arr, dtype=float) - self.mean_) / self.std_

    def inverse(self, arr):
        return np.asarray(arr, dtype=float) * self.std_ + self.mean_


def monotone_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """Map a 3-channel raw tensor to non-crossing [q10, q50, q90].

    raw[..., 0] = q50 (median, free), raw[..., 1] = up logit, raw[..., 2] = down
    logit. q90 = q50 + softplus(up), q10 = q50 - softplus(down). The result is
    monotone by construction.
    """
    base = raw[..., 0]
    up = F.softplus(raw[..., 1])
    down = F.softplus(raw[..., 2])
    q10 = base - down
    q50 = base
    q90 = base + up
    return torch.stack([q10, q50, q90], dim=-1)


class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al. 2022).

    Normalizes each instance over the time axis in 'norm' mode and inverts it
    in 'denorm' mode. Ablated, not assumed: instance stats are contaminated by
    price spikes, so this is only adopted if it wins on validation.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        self.mu = None
        self.sigma = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mu = x.mean(dim=1, keepdim=True).detach()
            self.sigma = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            out = (x - self.mu) / self.sigma
            if self.affine:
                out = out * self.gamma + self.beta
            return out
        if mode == "denorm":
            out = x
            if self.affine:
                out = (out - self.beta) / (self.gamma + self.eps * self.eps)
            return out * self.sigma + self.mu
        raise ValueError(f"unknown mode: {mode}")
