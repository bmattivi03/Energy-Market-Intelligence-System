"""Agent training, evaluation, and checkpointing for Module C.

Flat 3-level structure mirroring module_b.models:

* :class:`BaseAgent` - ABC enforcing act / train / save / load.
* :class:`PPOAgent` / :class:`SACAgent` - thin stable-baselines3 wrappers.
* :func:`evaluate_agent` - standalone evaluation returning :class:`EvalResult`.
* :class:`TrainConfig` - all hyperparameters in one dataclass.
* :data:`AGENT_REGISTRY` + :func:`build_agent` - algo lookup by name.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np

from module_c.environment import BatteryEnv
from module_c.reward import CvarRewardShaper, compute_cvar


# ── result dataclass ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Summary of an evaluation run across multiple episodes.

    Mirrors :class:`~module_b.evaluation.DMResult` in structure: plain data,
    no behaviour.
    """
    mean_profit: float    # mean total episode profit (EUR)
    std_profit: float     # standard deviation across episodes
    mean_cvar: float      # CVaR(5%) across all per-step returns
    sharpe: float         # mean_profit / std_profit (0.0 when std == 0)
    n_episodes: int


# ── training config ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All hyperparameters for one training run."""
    algo: str = "PPO"
    total_timesteps: int = 100_000
    lambda_risk: float = 0.1          # forwarded to CvarRewardShaper
    n_envs: int = 4                   # PPO uses parallel envs; SAC ignores this
    seed: int = 42
    checkpoint_dir: Path = field(
        default_factory=lambda: Path("checkpoints/module_c")
    )
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    policy_kwargs: Optional[dict] = None   # passed verbatim to SB3 constructor


# ── base ABC ─────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base for RL trading agents.

    Mirrors :class:`~module_b.models.BaseQuantileForecaster` one-for-one:
    ``act`` <-> ``predict_quantiles``, ``train`` <-> ``fit``.
    """

    name: str

    def __init__(self, *, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        """Return an action array for a single observation."""

    @abstractmethod
    def train(self, env: BatteryEnv, config: TrainConfig) -> "BaseAgent":
        """Fit the agent. Returns self for chaining."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model weights and metadata to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseAgent":
        """Load a previously saved agent."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ── PPO ──────────────────────────────────────────────────────────────────────

class PPOAgent(BaseAgent):
    """stable-baselines3 PPO with MlpPolicy.

    On-policy: benefits from ``n_envs`` parallel environments for variance
    reduction. More stable than SAC but less sample-efficient.
    """

    def __init__(self, *, name: Optional[str] = None) -> None:
        super().__init__(name=name or "ppo")
        self._model = None

    def train(self, env: BatteryEnv, config: TrainConfig) -> "PPOAgent":
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        # Apply lambda_risk from config to the caller's env too (keeps them in sync).
        env._shaper.lambda_risk = config.lambda_risk

        # PPO needs n_envs *independent* instances - lambda: env would hand the same
        # object to every worker, causing all of them to race over shared state.
        _df = env._df
        _cfg = env._cfg
        _shaper_kw = dict(
            lambda_risk=config.lambda_risk,
            alpha=env._shaper.alpha,
            window=env._shaper.window,
            min_history=env._shaper.min_history,
        )

        def _factory() -> BatteryEnv:
            return BatteryEnv(
                _df,
                config=_cfg,
                reward_shaper=CvarRewardShaper(**_shaper_kw),
                seed=config.seed,
            )

        vec_env = make_vec_env(_factory, n_envs=config.n_envs, seed=config.seed)
        self._model = PPO(
            "MlpPolicy",
            vec_env,
            seed=config.seed,
            verbose=0,
            # MlpPolicy runs faster on CPU than MPS/GPU, and SB3 on Apple MPS
            # segfaults during training - pin CPU.
            device="cpu",
            **(config.policy_kwargs or {}),
        )
        self._model.learn(total_timesteps=config.total_timesteps)
        return self

    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("must call train() before act()")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "model"))
        (path / "meta.json").write_text(
            json.dumps({"algo": "PPO", "name": self.name})
        )

    @classmethod
    def load(cls, path: Path) -> "PPOAgent":
        from stable_baselines3 import PPO

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        agent = cls(name=meta.get("name"))
        agent._model = PPO.load(str(path / "model"))
        return agent


# ── SAC ──────────────────────────────────────────────────────────────────────

class SACAgent(BaseAgent):
    """stable-baselines3 SAC with MlpPolicy.

    Off-policy: uses a replay buffer, so n_envs=1 is standard. Better suited
    to continuous action spaces than PPO; more sample-efficient in practice.
    """

    def __init__(self, *, name: Optional[str] = None) -> None:
        super().__init__(name=name or "sac")
        self._model = None

    def train(self, env: BatteryEnv, config: TrainConfig) -> "SACAgent":
        from stable_baselines3 import SAC

        env._shaper.lambda_risk = config.lambda_risk

        self._model = SAC(
            "MlpPolicy",
            env,
            seed=config.seed,
            verbose=0,
            # See PPOAgent: pin CPU (SB3 MlpPolicy is faster on CPU and MPS
            # training segfaults).
            device="cpu",
            **(config.policy_kwargs or {}),
        )
        self._model.learn(total_timesteps=config.total_timesteps)
        return self

    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("must call train() before act()")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "model"))
        (path / "meta.json").write_text(
            json.dumps({"algo": "SAC", "name": self.name})
        )

    @classmethod
    def load(cls, path: Path) -> "SACAgent":
        from stable_baselines3 import SAC

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        agent = cls(name=meta.get("name"))
        agent._model = SAC.load(str(path / "model"))
        return agent


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_agent(
    agent: BaseAgent,
    env: BatteryEnv,
    n_episodes: int = 10,
    *,
    cvar_alpha: float = 0.05,
) -> EvalResult:
    """Run ``n_episodes`` deterministic episodes and aggregate performance.

    Mirrors the standalone-function pattern of
    :func:`~module_b.evaluation.segment_metrics`.
    """
    episode_profits: list[float] = []
    all_step_profits: list[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_profit = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            step_profit = float(info["step_profit"])
            ep_profit += step_profit
            all_step_profits.append(step_profit)
        episode_profits.append(ep_profit)

    profits = np.array(episode_profits, dtype=np.float64)
    mean_p = float(profits.mean())
    std_p = float(profits.std())
    sharpe = mean_p / std_p if std_p > 0 else 0.0
    mean_cvar = compute_cvar(all_step_profits, alpha=cvar_alpha)

    return EvalResult(
        mean_profit=mean_p,
        std_profit=std_p,
        mean_cvar=mean_cvar,
        sharpe=sharpe,
        n_episodes=n_episodes,
    )


# ── registry ─────────────────────────────────────────────────────────────────

AGENT_REGISTRY: dict[str, Type[BaseAgent]] = {
    "PPO": PPOAgent,
    "SAC": SACAgent,
}


def build_agent(config: TrainConfig) -> BaseAgent:
    """Instantiate the agent named in ``config.algo``."""
    cls = AGENT_REGISTRY.get(config.algo.upper())
    if cls is None:
        raise ValueError(
            f"Unknown algo {config.algo!r}; available: {list(AGENT_REGISTRY)}"
        )
    return cls()
