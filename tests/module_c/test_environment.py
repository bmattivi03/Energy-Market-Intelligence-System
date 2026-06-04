"""Tests for module_c.environment.BatteryEnv.

Exercises the real Gymnasium contract: observation dimensionality (78 base,
82 with the 4 load extras), the 5-tuple step return, SoC bounds, the
charge/discharge action-sign convention, episode truncation, and the
zero-fill behaviour of `_validate_columns` for missing forecast columns.
"""

import numpy as np
import pytest

from module_c.environment import (
    BatteryEnv,
    BatteryEnvConfig,
    _OBS_DIM,
)

from .conftest import LOAD_EXTRA_COLS


# ── observation dimensionality ───────────────────────────────────────────────

def test_obs_dim_is_78_base(price_df):
    env = BatteryEnv(price_df, seed=0)
    assert _OBS_DIM == 78
    obs, info = env.reset(seed=0)
    assert obs.shape == (78,)
    assert env.observation_space.shape == (78,)
    assert info == {}


def test_obs_dim_extends_with_extra_cols(price_df_with_load):
    cfg = BatteryEnvConfig(extra_cols=LOAD_EXTRA_COLS)
    env = BatteryEnv(price_df_with_load, config=cfg, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (82,)
    assert env.observation_space.shape == (82,)


def test_reset_obs_within_observation_space(price_df):
    env = BatteryEnv(price_df, seed=1)
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    # soc_norm (component 0) must lie in [0, 1].
    assert 0.0 <= obs[0] <= 1.0


def test_reset_is_deterministic_under_seed(price_df):
    env_a = BatteryEnv(price_df, seed=123)
    env_b = BatteryEnv(price_df, seed=999)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_allclose(obs_a, obs_b)


# ── step contract ────────────────────────────────────────────────────────────

def test_step_returns_five_tuple_with_finite_reward(price_df):
    env = BatteryEnv(price_df, seed=2)
    env.reset(seed=2)
    out = env.step(np.array([0.3], dtype=np.float32))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert terminated is False  # env never terminates, only truncates
    for key in ("price", "power_mw_actual", "soc_mwh", "step_profit", "episode_cvar"):
        assert key in info
    assert np.isfinite(info["step_profit"])


def test_action_clipping_does_not_raise(price_df):
    env = BatteryEnv(price_df, seed=3)
    env.reset(seed=3)
    # Out-of-range action is clipped internally to [-1, 1].
    obs, reward, _, _, info = env.step(np.array([5.0], dtype=np.float32))
    assert np.isfinite(reward)
    assert np.isfinite(info["power_mw_actual"])


# ── SoC bounds ───────────────────────────────────────────────────────────────

def test_soc_stays_within_bounds_over_random_episode(price_df):
    cap = 100.0
    cfg = BatteryEnvConfig(capacity_mwh=cap, episode_len=24)
    env = BatteryEnv(price_df, config=cfg, seed=4)
    env.reset(seed=4)
    rng = np.random.default_rng(4)
    truncated = False
    while not truncated:
        a = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
        _, _, _, truncated, info = env.step(a)
        assert 0.0 - 1e-6 <= info["soc_mwh"] <= cap + 1e-6


def test_soc_clamps_at_zero_when_forced_to_discharge(price_df):
    # Start with a tiny SoC, then keep discharging (positive action).
    cfg = BatteryEnvConfig(capacity_mwh=100.0, initial_soc_range=(0.0, 0.0), episode_len=24)
    env = BatteryEnv(price_df, config=cfg, seed=5)
    env.reset(seed=5)
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(np.array([1.0], dtype=np.float32))
        assert info["soc_mwh"] >= -1e-6
    # An empty battery cannot deliver energy: SoC pinned at zero.
    assert info["soc_mwh"] == pytest.approx(0.0, abs=1e-6)


def test_soc_clamps_at_capacity_when_forced_to_charge(price_df):
    cap = 100.0
    cfg = BatteryEnvConfig(capacity_mwh=cap, initial_soc_range=(1.0, 1.0), episode_len=24)
    env = BatteryEnv(price_df, config=cfg, seed=6)
    env.reset(seed=6)
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(np.array([-1.0], dtype=np.float32))
        assert info["soc_mwh"] <= cap + 1e-6
    assert info["soc_mwh"] == pytest.approx(cap, abs=1e-6)


# ── action sign convention ───────────────────────────────────────────────────

def test_positive_action_discharges_soc_decreases(price_df):
    cfg = BatteryEnvConfig(initial_soc_range=(0.5, 0.5))
    env = BatteryEnv(price_df, config=cfg, seed=7)
    env.reset(seed=7)
    soc_before = env._soc
    _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    # Positive action = discharge (sell) → SoC falls, power_actual positive.
    assert info["soc_mwh"] < soc_before
    assert info["power_mw_actual"] > 0.0
    assert info["power_mw_requested"] > 0.0


def test_negative_action_charges_soc_increases(price_df):
    cfg = BatteryEnvConfig(initial_soc_range=(0.5, 0.5))
    env = BatteryEnv(price_df, config=cfg, seed=8)
    env.reset(seed=8)
    soc_before = env._soc
    _, _, _, _, info = env.step(np.array([-0.5], dtype=np.float32))
    # Negative action = charge (buy) → SoC rises, power_actual negative.
    assert info["soc_mwh"] > soc_before
    assert info["power_mw_actual"] < 0.0
    assert info["power_mw_requested"] < 0.0


def test_discharge_profit_positive_charge_profit_negative(price_df):
    cfg = BatteryEnvConfig(initial_soc_range=(0.5, 0.5))
    # Prices in the fixture are strictly positive, so sign of step_profit
    # follows sign of power_actual.
    env = BatteryEnv(price_df, config=cfg, seed=9)
    env.reset(seed=9)
    _, _, _, _, info_dis = env.step(np.array([0.8], dtype=np.float32))
    assert info_dis["step_profit"] > 0.0

    env.reset(seed=9)
    _, _, _, _, info_chg = env.step(np.array([-0.8], dtype=np.float32))
    assert info_chg["step_profit"] < 0.0


# ── episode truncation ───────────────────────────────────────────────────────

def test_episode_truncates_at_episode_len(price_df):
    ep_len = 12
    cfg = BatteryEnvConfig(episode_len=ep_len)
    env = BatteryEnv(price_df, config=cfg, seed=10)
    env.reset(seed=10)
    truncated = False
    steps = 0
    while not truncated:
        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        steps += 1
        assert terminated is False
        assert steps <= ep_len  # must not overrun
    assert steps == ep_len


def test_terminal_obs_is_zero_vector(price_df):
    cfg = BatteryEnvConfig(episode_len=3)
    env = BatteryEnv(price_df, config=cfg, seed=11)
    env.reset(seed=11)
    truncated = False
    obs = None
    while not truncated:
        obs, _, _, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    # On truncation the env returns an all-zero observation of the right shape.
    assert obs.shape == env.observation_space.shape
    np.testing.assert_array_equal(obs, np.zeros(env.observation_space.shape[0], dtype=np.float32))


# ── missing-column validation ────────────────────────────────────────────────

def test_missing_forecast_columns_are_zero_filled(price_df):
    # Drop every forecast column: raw / no-forecast mode.
    raw = price_df[["price"]].copy()
    env = BatteryEnv(raw, seed=12)
    obs, _ = env.reset(seed=12)
    # Still produces a full 78-dim observation...
    assert obs.shape == (78,)
    # ...and the 72 forecast slots (indices 6..78) are z-scored zeros, i.e.
    # (0 - price_mean) / price_std - all identical.
    forecast_block = obs[6:78]
    assert np.all(forecast_block == forecast_block[0])


def test_missing_price_column_raises():
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    bad = pd.DataFrame({"not_price": np.ones(24)}, index=idx)
    with pytest.raises(ValueError, match="missing"):
        BatteryEnv(bad)
