"""Central configuration — single source of truth for module hyperparameters.

Frozen dataclasses collect the defaults that were previously scattered across
each module's ``argparse`` and constructor defaults. CLIs read their argparse
defaults from these classes so there is exactly one place to change a value.

Split bounds and the forecast HORIZON / LOOKBACK are *not* re-hardcoded here —
they are imported from ``data.schemas`` / ``data.splits`` which remain the
authoritative source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from data.schemas import SPLIT_BOUNDS
from data.splits import HORIZON, LOOKBACK


@dataclass(frozen=True)
class DataConfig:
    """Forecast geometry and split bounds (references, not redefinitions)."""

    horizon: int = HORIZON          # 24h day-ahead horizon (from data.splits)
    lookback: int = LOOKBACK        # 168h input window (from data.splits)
    # Authoritative (start, end) UTC bounds per split, from data.schemas.
    split_bounds: dict = field(default_factory=lambda: dict(SPLIT_BOUNDS))


@dataclass(frozen=True)
class ModuleAConfig:
    """Multi-scale LSTM load forecaster (Module A) defaults.

    Mirrors ``module_a.train`` argparse and ``module_a.model`` constructor.
    """

    seed: int = 42
    epochs: int = 200
    patience: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    hidden: int = 128
    dropout: float = 0.2
    num_layers_short: int = 2
    num_layers_long: int = 1


@dataclass(frozen=True)
class ModuleBConfig:
    """CatBoost + CQR price forecaster (Module B) defaults.

    Mirrors ``module_b.train`` argparse / B6 notebook values.
    """

    seed: int = 42
    iterations: int = 500
    early_stopping: int = 30
    lr: float = 0.05
    cqr_alpha: float = 0.20
    # Production feature bundles (no load_quantiles by default — v1 standalone).
    bundles: tuple[str, ...] = (
        "calendar", "lags", "fundamentals", "spike", "regime", "weather",
    )


@dataclass(frozen=True)
class ModuleCConfig:
    """RL battery-dispatch (Module C) defaults.

    Mirrors ``module_c.run`` / ``module_c.environment`` / ``module_c.train``.
    """

    seed: int = 42
    # Battery physical parameters (module_c.environment).
    capacity_mwh: float = 100.0
    max_power_mw: float = 50.0
    efficiency: float = 0.90
    episode_len: int = 24
    # Training / evaluation budget (module_c.run, module_c.train).
    total_timesteps: int = 200_000
    n_eval_episodes: int = 200
    lambda_risk: float = 0.1


# Convenience singletons — import these for read-only access.
DATA = DataConfig()
MODULE_A = ModuleAConfig()
MODULE_B = ModuleBConfig()
MODULE_C = ModuleCConfig()
