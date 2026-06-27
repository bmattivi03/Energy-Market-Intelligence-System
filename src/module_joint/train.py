"""Training loop, early stopping, and CLI for the joint model.

MPS hygiene mirrors src/preprocessing/impute.py: num_workers=0, pin_memory only
on CUDA, weights_only loads. The validation metric is the total normalized pinball
(load + price), which is what early stopping monitors.
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.loaders import load_split

from .config import QUANTILES, JointConfig, select_device, set_seed
from .data import build_windows
from .losses import DWA, UncertaintyWeighting, joint_loss, multi_quantile_pinball
from .model import JointForecasterEstimator

WEATHER_SLICE = slice(8, 28)  # future channels: [cal(8), weather(20), loadq(3)]


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, value: float, model) -> bool:
        """Return True if training should stop."""
        if value < self.best - 1e-6:
            self.best = value
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def _to_dataset(wa) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(wa.past),
        torch.from_numpy(wa.future),
        torch.from_numpy(wa.y["load"]),
        torch.from_numpy(wa.y["price"]),
        torch.from_numpy(wa.y["residual_load"]),
        torch.from_numpy(wa.y["renewables"]),
        torch.from_numpy(wa.y["spike"]),
    )


def _unpack(batch, device):
    past, future, yl, yp, yr, yren, ysp = [b.to(device) for b in batch]
    targets = {
        "load": yl,
        "price": yp,
        "residual_load": yr,
        "renewables": yren,
        "spike": ysp,
    }
    return past, future, targets


def _weather_noise(future: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return future
    h = future.shape[1]
    grow = torch.sqrt(torch.arange(1, h + 1, device=future.device).float()).view(1, h, 1)
    noise = torch.randn_like(future[:, :, WEATHER_SLICE]) * std * grow
    out = future.clone()
    out[:, :, WEATHER_SLICE] = out[:, :, WEATHER_SLICE] + noise
    return out


@torch.no_grad()
def _val_pinball(model, val_ds, device) -> float:
    model.eval()
    loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    tot, n = 0.0, 0
    for batch in loader:
        past, future, targets = _unpack(batch, device)
        out = model(past, future)
        pl = multi_quantile_pinball(out["load"], targets["load"], QUANTILES)
        pp = multi_quantile_pinball(out["price"], targets["price"], QUANTILES)
        bs = past.shape[0]
        tot += float(pl + pp) * bs
        n += bs
    return tot / max(n, 1)


def train_one(cfg, train_df, val_df, load_quantiles, device=None, estimator=None):
    set_seed(cfg.seed)
    device = device or select_device()

    train_wa, scalers = build_windows(
        train_df, load_quantiles=load_quantiles, fit=True,
        use_load_quantiles=cfg.use_load_quantiles, fundamentals=cfg.fundamentals,
        anchor_residual=cfg.anchor_residual, anchor_col=cfg.anchor_col,
        lookback=cfg.lookback,
    )
    val_wa, _ = build_windows(
        val_df, load_quantiles=load_quantiles, scalers=scalers, fit=False,
        use_load_quantiles=cfg.use_load_quantiles, fundamentals=cfg.fundamentals,
        anchor_residual=cfg.anchor_residual, anchor_col=cfg.anchor_col,
        restrict_to=val_df.index, lookback=cfg.lookback,
    )

    estimator = estimator or JointForecasterEstimator(cfg)
    estimator.setup_from_windows(train_wa, scalers)
    estimator.device = device
    model = estimator.model.to(device)

    train_ds = _to_dataset(train_wa)
    val_ds = _to_dataset(val_wa)
    pin = device.type == "cuda"
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=pin
    )

    params = list(model.parameters())
    weighter = None
    if cfg.loss_weighting == "uncertainty":
        weighter = UncertaintyWeighting(2).to(device)
        params += list(weighter.parameters())
    elif cfg.loss_weighting == "dwa":
        weighter = DWA(2)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    stopper = EarlyStopping(cfg.patience)

    history = {"val": []}
    for _epoch in range(cfg.max_epochs):
        model.train()
        for batch in loader:
            past, future, targets = _unpack(batch, device)
            future = _weather_noise(future, cfg.weather_noise_std)
            opt.zero_grad()
            out = model(past, future)
            loss, _ = joint_loss(out, targets, cfg, weighter)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
        sched.step()
        v = _val_pinball(model, val_ds, device)
        history["val"].append(v)
        if stopper.step(v, model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return estimator, history


def main():
    ap = argparse.ArgumentParser(description="Train the joint load+price model.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="tide", choices=["tide", "nhitsx", "dlinear"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--out", default="checkpoints/module_joint/model")
    args = ap.parse_args()

    cfg = JointConfig(seed=args.seed, backbone=args.backbone, max_epochs=args.epochs)
    import pandas as pd  # noqa: F401  (kept explicit for clarity)

    load_quantiles = pd.read_parquet("data/module_a/load_quantiles.parquet")
    train_df = load_split("train")
    val_df = load_split("val")
    device = select_device(force_cpu=args.force_cpu)
    est, history = train_one(cfg, train_df, val_df, load_quantiles, device=device)
    est.save(args.out)
    print(f"best val pinball: {min(history['val']):.5f}  saved to {args.out}")


if __name__ == "__main__":
    main()
