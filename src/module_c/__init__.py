"""Module C - risk-aware RL battery trading agent.

Flat 3-file layout mirroring module_b:

* :mod:`module_c.environment` - Gymnasium BESS env + BatteryEnvConfig.
* :mod:`module_c.reward` - compute_profit, compute_cvar, CvarRewardShaper.
* :mod:`module_c.train` - BaseAgent ABC, PPO/SAC wrappers, TrainConfig,
  EvalResult, evaluate_agent, AGENT_REGISTRY, build_agent.

Notebooks in ``notebooks/module_c/`` drive experiments. This package only
provides reusable building blocks.
"""

from module_c.environment import BatteryEnv, BatteryEnvConfig
from module_c.reward import CvarRewardShaper, compute_cvar, compute_profit
from module_c.train import (
    AGENT_REGISTRY,
    BaseAgent,
    EvalResult,
    PPOAgent,
    SACAgent,
    TrainConfig,
    build_agent,
    evaluate_agent,
)

__all__ = [
    # environment
    "BatteryEnv",
    "BatteryEnvConfig",
    # reward
    "compute_profit",
    "compute_cvar",
    "CvarRewardShaper",
    # agents + training
    "BaseAgent",
    "PPOAgent",
    "SACAgent",
    "TrainConfig",
    "EvalResult",
    "AGENT_REGISTRY",
    "build_agent",
    "evaluate_agent",
]
