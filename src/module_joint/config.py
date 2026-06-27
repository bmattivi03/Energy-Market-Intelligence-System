"""Configuration, constants, seeding, and device selection for module_joint.

This is the single source of hyperparameters and ablation flags for the joint
multi-task forecaster. Mirrors the reproducibility and device conventions used
by src/preprocessing/impute.py.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

HORIZON = 24
LOOKBACK = 168
QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_COLS = ("q10", "q50", "q90")
ALPHA = 0.20

# Target names (kept here to avoid a hard import cycle; validated against schemas in tests).
LOAD = "load"
PRICE = "price"


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (cpu + cuda) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(force_cpu: bool = False) -> torch.device:
    """Pick the best available device: CUDA, then MPS, then CPU.

    EMIS_FORCE_CPU=1 or force_cpu forces CPU (used on flaky MPS kernels).
    """
    if force_cpu or os.environ.get("EMIS_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class JointConfig:
    """All hyperparameters and ablation flags for the joint model.

    Ablation flags (use_load_quantiles, load_to_price, aux_*, revin,
    loss_weighting) drive the Phase 2 experiments without code changes.
    """

    # windowing
    lookback: int = LOOKBACK
    horizon: int = HORIZON

    # architecture
    hidden: int = 256
    enc_layers: int = 2
    dec_layers: int = 2
    dropout: float = 0.2
    backbone: str = "tide"  # tide | nhitsx | dlinear

    # multi-task structure (ablation flags)
    use_load_quantiles: bool = True  # Module A external load quantiles as future input
    load_to_price: bool = True  # internal directed load->price pathway
    detach_load_to_price: bool = False
    aux_residual_load: bool = True
    aux_spike: bool = True
    aux_renewables: bool = True
    revin: bool = False

    # feature parity + fuel-cost anchor (Phase: beat-CatBoost iteration)
    fundamentals: bool = False  # add Module B engineered features to the encoder input
    anchor_residual: bool = False  # predict price as deviation from the fuel-cost anchor
    anchor_col: str = "clean_spark_anchor"

    # loss weighting
    loss_weighting: str = "dwa"  # dwa | uncertainty | fixed
    task_weights: dict = field(default_factory=lambda: {"load": 1.0, "price": 1.0})
    aux_weight: float = 0.3
    spike_weight: float = 0.5
    horizon_weight_h1_6: float = 1.5  # emphasize the headline h1-6 segment

    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 12
    grad_clip: float = 1.0

    # known-future weather train/serve mismatch mitigation
    weather_noise_std: float = 0.0  # horizon-growing gaussian noise on weather (train only)

    seed: int = 42
