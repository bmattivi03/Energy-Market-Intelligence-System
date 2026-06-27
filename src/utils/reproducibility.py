"""Canonical reproducibility helpers shared across all modules.

Single source of truth for RNG seeding and device selection. Lifted from the
imputation pipeline (``preprocessing.impute``) — the most complete versions —
so Module A / B / C and the imputer all behave identically.

Both functions import ``torch`` lazily so importing this module stays cheap and
torch-free for callers that only need ``set_seed`` without a CUDA/MPS probe.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every RNG that affects reproducibility.

    Seeds ``random``, ``numpy`` and ``torch`` (cpu + ``cuda.manual_seed_all`` +
    mps when available), and forces cuDNN into deterministic mode. The cuDNN
    flags are guarded so this works on machines without CUDA.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(prefer: str | None = None) -> str:
    """Return ``"cuda" | "mps" | "cpu"``. ``prefer`` forces a specific device
    only if it is actually available. ``EMIS_FORCE_CPU=1`` forces CPU."""
    import torch

    if os.environ.get("EMIS_FORCE_CPU") == "1":
        return "cpu"
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
