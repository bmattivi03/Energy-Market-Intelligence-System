"""The joint multi-task model and its sklearn-style estimator.

JointForecaster: shared encoder -> per-task decoders -> heads, with a directed
load->price pathway and auxiliary heads. JointForecasterEstimator wraps it with
fit / predict_quantiles / save / load and emits the long-form forecast contract
(dict[target] -> DataFrame indexed by (origin_ts, horizon_h), columns q10/q50/q90,
load in MW and price in EUR/MWh).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .backbones import ResidualMLPBlock, build_backbone
from .config import QUANTILE_COLS, JointConfig, select_device
from .data import Scalers, build_windows
from .heads import MonotoneQuantileHead, RegressionHead, SpikeHead
from .transforms import RevIN, inv_asinh


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden, dropout) for _ in range(max(0, layers - 1))]
        )

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return x


class JointForecaster(nn.Module):
    def __init__(self, cfg: JointConfig, n_past_feat: int, n_future_feat: int):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.horizon
        self.revin = RevIN(n_past_feat) if cfg.revin else None
        self.backbone = build_backbone(cfg.backbone, n_past_feat, cfg.lookback, cfg)
        z = self.backbone.out_dim

        self.fdim = max(8, cfg.hidden // 8)
        self.fut_embed = nn.Sequential(nn.Linear(n_future_feat, self.fdim), nn.ReLU())
        fut_flat = self.h * self.fdim

        self.load_decoder = _MLP(z + fut_flat, cfg.hidden, cfg.dec_layers, cfg.dropout)
        self.load_head = MonotoneQuantileHead(cfg.hidden, self.h)

        price_extra = 3 * self.h if cfg.load_to_price else 0
        self.price_decoder = _MLP(
            z + fut_flat + price_extra, cfg.hidden, cfg.dec_layers, cfg.dropout
        )
        self.price_head = MonotoneQuantileHead(cfg.hidden, self.h)

        self.resid_head = RegressionHead(z, self.h, 1) if cfg.aux_residual_load else None
        self.renew_head = RegressionHead(z, self.h, 2) if cfg.aux_renewables else None
        self.spike_head = SpikeHead(z, self.h) if cfg.aux_spike else None

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> dict:
        if self.revin is not None:
            past = self.revin(past, "norm")
        z = self.backbone(past)
        f = self.fut_embed(future).flatten(1)

        load_q = self.load_head(self.load_decoder(torch.cat([z, f], dim=-1)))
        price_in = [z, f]
        if self.cfg.load_to_price:
            lq = load_q.detach() if self.cfg.detach_load_to_price else load_q
            price_in.append(lq.flatten(1))
        price_q = self.price_head(self.price_decoder(torch.cat(price_in, dim=-1)))

        out = {"load": load_q, "price": price_q}
        if self.resid_head is not None:
            out["residual_load"] = self.resid_head(z)
        if self.renew_head is not None:
            out["renewables"] = self.renew_head(z)
        if self.spike_head is not None:
            out["spike"] = self.spike_head(z)
        return out


class JointForecasterEstimator:
    """Fit/predict wrapper that owns the model + scalers and the IO contract."""

    def __init__(self, cfg: JointConfig | None = None):
        self.cfg = cfg or JointConfig()
        self.model: JointForecaster | None = None
        self.scalers: Scalers | None = None
        self.n_past_feat: int | None = None
        self.n_future_feat: int | None = None
        self.device = None

    def _build(self, n_past_feat: int, n_future_feat: int):
        self.n_past_feat = n_past_feat
        self.n_future_feat = n_future_feat
        self.model = JointForecaster(self.cfg, n_past_feat, n_future_feat)
        return self.model

    def setup_from_windows(self, wa, scalers: Scalers):
        """Initialize scalers + model dimensions from a fitted WindowArrays."""
        self.scalers = scalers
        self._build(wa.past.shape[2], wa.future.shape[2])
        return self

    def fit(self, train_df, val_df, load_quantiles, device=None):
        from .train import train_one  # lazy import to avoid a cycle

        est, _ = train_one(
            self.cfg, train_df, val_df, load_quantiles, device=device, estimator=self
        )
        return est

    @torch.no_grad()
    def predict_quantiles(self, df, load_quantiles, *, restrict_to=None) -> dict:
        assert self.model is not None and self.scalers is not None
        wa, _ = build_windows(
            df,
            load_quantiles=load_quantiles,
            scalers=self.scalers,
            fit=False,
            use_load_quantiles=self.cfg.use_load_quantiles,
            fundamentals=self.cfg.fundamentals,
            anchor_residual=self.cfg.anchor_residual,
            anchor_col=self.cfg.anchor_col,
            restrict_to=restrict_to,
            lookback=self.cfg.lookback,
        )
        device = self.device or select_device()
        self.model.to(device).eval()
        past = torch.from_numpy(wa.past).to(device)
        future = torch.from_numpy(wa.future).to(device)

        loads, prices = [], []
        bs = 512
        for i in range(0, past.shape[0], bs):
            out = self.model(past[i : i + bs], future[i : i + bs])
            loads.append(out["load"].cpu().numpy())
            prices.append(out["price"].cpu().numpy())
        load_s = np.concatenate(loads, axis=0)  # (N,H,3) scaled
        price_s = np.concatenate(prices, axis=0)

        load_mw = self.scalers.load.inverse(load_s)
        price_eur = inv_asinh(self.scalers.price.inverse(price_s))
        if self.scalers.anchor_residual:  # add the fuel-cost anchor (per origin) back
            price_eur = price_eur + wa.anchor[:, None, None]
        return {
            "load": self._to_long(load_mw, wa.origin_ts),
            "price": self._to_long(price_eur, wa.origin_ts),
        }

    def _to_long(self, arr: np.ndarray, origin_ts: np.ndarray) -> pd.DataFrame:
        n, h, _ = arr.shape
        origins = np.repeat(origin_ts, h)
        horizons = np.tile(np.arange(1, h + 1), n)
        flat = arr.reshape(n * h, 3)
        idx = pd.MultiIndex.from_arrays([origins, horizons], names=["origin_ts", "horizon_h"])
        return pd.DataFrame(flat, index=idx, columns=list(QUANTILE_COLS))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path.with_suffix(".pt"))
        with open(path.with_suffix(".scalers.pkl"), "wb") as f:
            pickle.dump(
                {
                    "cfg": self.cfg,
                    "scalers": self.scalers,
                    "n_past_feat": self.n_past_feat,
                    "n_future_feat": self.n_future_feat,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path.with_suffix(".scalers.pkl"), "rb") as f:
            meta = pickle.load(f)
        est = cls(meta["cfg"])
        est.scalers = meta["scalers"]
        est._build(meta["n_past_feat"], meta["n_future_feat"])
        state = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        est.model.load_state_dict(state)
        est.device = torch.device("cpu")
        return est
