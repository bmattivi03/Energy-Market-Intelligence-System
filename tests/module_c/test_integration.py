"""End-to-end Module C integration tests — no trained checkpoints, no network.

1. A full random-policy episode driving a real `BatteryEnv` built from the
   synthetic `price_df`, asserting the Gymnasium contract end to end.
2. A minimal agent smoke test: `build_agent` / `PPOAgent` construct, a tiny
   training budget runs, and `.act()` returns a valid-shape action.

Training is capped at a handful of timesteps so the suite stays well under a
minute. stable_baselines3 is import-skipped if unavailable.
"""

import numpy as np
import pytest

from module_c.environment import BatteryEnv, BatteryEnvConfig
from module_c.train import AGENT_REGISTRY, PPOAgent, TrainConfig, build_agent


# ── random-policy full episode ───────────────────────────────────────────────

def test_random_policy_full_episode(price_df):
    cfg = BatteryEnvConfig(episode_len=24)
    env = BatteryEnv(price_df, config=cfg, seed=0)
    env.action_space.seed(0)

    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert info == {}

    n_steps = 0
    truncated = terminated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        assert env.action_space.contains(action)

        obs, reward, terminated, truncated, info = env.step(action)
        n_steps += 1

        # Obs always matches the declared observation space shape.
        assert obs.shape == env.observation_space.shape
        # Reward is always finite.
        assert np.isfinite(reward)
        # Info carries the dispatch diagnostics the evaluator relies on.
        assert "step_profit" in info and np.isfinite(info["step_profit"])
        assert "soc_mwh" in info
        assert 0.0 - 1e-6 <= info["soc_mwh"] <= cfg.capacity_mwh + 1e-6
        # Never terminates; only truncates.
        assert terminated is False
        assert n_steps <= cfg.episode_len

    # Episode ends exactly at episode_len via truncation.
    assert truncated is True
    assert n_steps == cfg.episode_len


def test_random_policy_episode_with_load_extras(price_df_with_load):
    from .conftest import LOAD_EXTRA_COLS

    cfg = BatteryEnvConfig(episode_len=12, extra_cols=LOAD_EXTRA_COLS)
    env = BatteryEnv(price_df_with_load, config=cfg, seed=1)
    env.action_space.seed(1)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (82,)

    truncated = False
    while not truncated:
        obs, reward, _, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (82,)
        assert np.isfinite(reward)


# ── agent construction smoke tests ───────────────────────────────────────────

def test_build_agent_returns_ppo():
    agent = build_agent(TrainConfig(algo="PPO"))
    assert isinstance(agent, PPOAgent)
    assert agent.name == "ppo"


def test_build_agent_registry_and_unknown_algo():
    assert set(AGENT_REGISTRY) == {"PPO", "SAC"}
    with pytest.raises(ValueError, match="Unknown algo"):
        build_agent(TrainConfig(algo="DQN"))


def test_act_before_train_raises():
    agent = PPOAgent()
    with pytest.raises(RuntimeError, match="train"):
        agent.act(np.zeros(78, dtype=np.float32))


def test_ppo_tiny_train_then_act(price_df):
    pytest.importorskip("stable_baselines3")

    cfg = BatteryEnvConfig(episode_len=24)
    env = BatteryEnv(price_df, config=cfg, seed=0)

    # Minimal budget: a handful of timesteps across 2 envs. NOT 200k.
    train_cfg = TrainConfig(
        algo="PPO",
        total_timesteps=16,
        n_envs=2,
        seed=0,
        policy_kwargs={"n_steps": 8, "batch_size": 8},
    )
    agent = build_agent(train_cfg)
    agent.train(env, train_cfg)

    obs, _ = env.reset(seed=0)
    action = agent.act(obs, deterministic=True)
    action = np.asarray(action, dtype=np.float32)
    # Action must fit the env's action space (shape (1,), within [-1, 1]).
    assert action.shape == env.action_space.shape
    assert env.action_space.contains(action.astype(np.float32))
