"""Battery storage Gymnasium environment for day-ahead electricity trading.

The agent controls a virtual BESS (Battery Energy Storage System) in the
Germany-Luxembourg electricity market. At each hourly step it decides how much
power to charge or discharge; the market clears at the day-ahead price.

Observation (78-dim flat vector):
    soc_norm            — state of charge in [0, 1], normalised from capacity
    hour_sin, hour_cos  — cyclical hour encoding (same as Module B add_calendar)
    dow_sin, dow_cos    — cyclical day-of-week encoding
    price_lag1          — last actual price (EUR/MWh), z-scored on train stats
    q10_h1..q10_h24    — Module B q10 forecasts, z-scored (same stats as price)
    q50_h1..q50_h24    — Module B q50 forecasts, z-scored
    q90_h1..q90_h24    — Module B q90 forecasts, z-scored
    Total: 1 + 4 + 1 + 72 = 78 dimensions.

Action (continuous scalar):
    Box(-1, 1, shape=(1,)) linearly mapped to [-max_power_mw, +max_power_mw].
    Positive = discharge (sell energy to grid).
    Negative = charge (buy energy from grid).

Physical constraints:
    - SoC clipped to [0, capacity_mwh] after each energy delta.
    - Actual delivered power is back-calculated from the clipped delta, so the
      agent reads the true execution in info["power_mw_actual"].
    - Round-trip efficiency eta: charging stores E/sqrt(eta), discharging
      delivers E*sqrt(eta) to the grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym  # type: ignore[no-redef]
    from gym import spaces  # type: ignore[no-redef]

from module_c.reward import CvarRewardShaper


# ── physical defaults ────────────────────────────────────────────────────────

CAPACITY_MWH: float = 100.0
MAX_POWER_MW: float = 50.0
EFFICIENCY: float = 0.90          # round-trip (one-way ≈ sqrt(0.9) ≈ 0.949)
DT_H: float = 1.0                 # hourly timestep

_SQRT_EFF = math.sqrt(EFFICIENCY)

_QUANTILE_LABELS = ("q10", "q50", "q90")
_HORIZONS = 24
_FORECAST_DIMS = len(_QUANTILE_LABELS) * _HORIZONS   # 72
_OBS_DIM = 1 + 4 + 1 + _FORECAST_DIMS               # 78


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class BatteryEnvConfig:
    """Physical and episode-level hyperparameters."""
    capacity_mwh: float = CAPACITY_MWH
    max_power_mw: float = MAX_POWER_MW
    efficiency: float = EFFICIENCY
    episode_len: int = 24
    initial_soc_range: tuple[float, float] = (0.2, 0.8)  # fraction of capacity
    extra_cols: tuple = ()  # additional observation columns beyond the base 78


# ── environment ──────────────────────────────────────────────────────────────

class BatteryEnv(gym.Env):
    """Gymnasium BESS environment for day-ahead price arbitrage.

    Parameters
    ----------
    price_df:
        Time-indexed DataFrame (hourly UTC) with columns:
        - ``price``              actual clearing price (EUR/MWh)
        - ``q10_h1`` .. ``q90_h24``  Module B quantile forecasts
    reward_shaper:
        :class:`~module_c.reward.CvarRewardShaper` instance. A fresh default
        is created if omitted.
    config:
        Battery physical parameters and episode settings.
    seed:
        RNG seed for episode-start sampling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_df: pd.DataFrame,
        *,
        reward_shaper: Optional[CvarRewardShaper] = None,
        config: Optional[BatteryEnvConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._df = price_df.copy()
        self._cfg = config or BatteryEnvConfig()
        self._shaper = reward_shaper or CvarRewardShaper()

        self._validate_columns()

        # z-score statistics fitted on the provided data.
        # Caller should pass train-split data to avoid leakage.
        self._price_mean = float(self._df["price"].mean())
        self._price_std = float(self._df["price"].std()) or 1.0

        # Extra columns (e.g. load quantiles for B+A ablation): z-scored per col.
        self._extra_cols = list(self._cfg.extra_cols)
        self._extra_means = {c: float(self._df[c].mean()) for c in self._extra_cols}
        self._extra_stds  = {c: float(self._df[c].std()) or 1.0 for c in self._extra_cols}

        # Gymnasium spaces — obs_dim extends with extra cols.
        obs_dim = _OBS_DIM + len(self._extra_cols)
        obs_lo = np.full(obs_dim, -10.0, dtype=np.float32)
        obs_hi = np.full(obs_dim, 10.0, dtype=np.float32)
        obs_lo[0] = 0.0   # soc_norm lower bound
        obs_hi[0] = 1.0   # soc_norm upper bound
        self.observation_space = spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._start_idx: int = 0
        self._step_idx: int = 0
        self._soc: float = self._cfg.capacity_mwh * 0.5

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Leave enough runway for one full episode.
        max_start = len(self._df) - self._cfg.episode_len
        self._start_idx = int(self._rng.integers(0, max(1, max_start)))
        self._step_idx = 0

        lo, hi = self._cfg.initial_soc_range
        self._soc = float(self._rng.uniform(lo, hi) * self._cfg.capacity_mwh)
        self._shaper.reset()
        return self._make_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        row_idx = self._start_idx + self._step_idx
        price = float(self._df["price"].iloc[row_idx])
        sqrt_eff = math.sqrt(self._cfg.efficiency)

        # Decode continuous action to requested MW.
        power_requested = float(np.clip(action[0], -1.0, 1.0)) * self._cfg.max_power_mw

        # Energy delta with efficiency applied per direction.
        if power_requested >= 0:
            # Discharge: grid receives power_requested, battery loses more.
            energy_delta = -power_requested * DT_H / sqrt_eff
        else:
            # Charge: grid draws |power_requested|, battery gains less.
            energy_delta = -power_requested * DT_H * sqrt_eff

        # Enforce SoC bounds; back-calculate actual power from clipped delta.
        new_soc = float(
            np.clip(self._soc + energy_delta, 0.0, self._cfg.capacity_mwh)
        )
        actual_delta = new_soc - self._soc
        self._soc = new_soc

        if power_requested >= 0:
            power_actual = -actual_delta * sqrt_eff / DT_H
        else:
            power_actual = -actual_delta / (DT_H * sqrt_eff)

        step_profit = price * power_actual * DT_H
        reward = float(self._shaper.shape(step_profit))

        self._step_idx += 1
        truncated = self._step_idx >= self._cfg.episode_len
        obs = (
            np.zeros(self.observation_space.shape[0], dtype=np.float32)
            if truncated
            else self._make_obs()
        )
        info = {
            "price": price,
            "power_mw_requested": power_requested,
            "power_mw_actual": power_actual,
            "soc_mwh": self._soc,
            "step_profit": step_profit,
            "episode_cvar": self._shaper.current_cvar,
        }
        return obs, reward, False, truncated, info

    def render(self) -> None:
        row_idx = self._start_idx + self._step_idx
        ts = self._df.index[row_idx]
        print(
            f"[{ts}] step={self._step_idx:02d}  "
            f"SoC={self._soc:.1f}/{self._cfg.capacity_mwh:.0f}MWh  "
            f"CVaR={self._shaper.current_cvar:.2f}"
        )

    # ── private helpers ──────────────────────────────────────────────────────

    def _make_obs(self) -> np.ndarray:
        row_idx = self._start_idx + self._step_idx
        row = self._df.iloc[row_idx]
        ts = self._df.index[row_idx]

        soc_norm = self._soc / self._cfg.capacity_mwh
        hour_sin = math.sin(2 * math.pi * ts.hour / 24)
        hour_cos = math.cos(2 * math.pi * ts.hour / 24)
        dow_sin = math.sin(2 * math.pi * ts.dayofweek / 7)
        dow_cos = math.cos(2 * math.pi * ts.dayofweek / 7)
        price_z = (float(row["price"]) - self._price_mean) / self._price_std

        forecasts: list[float] = []
        for q_label in _QUANTILE_LABELS:
            for h in range(1, _HORIZONS + 1):
                col = f"{q_label}_h{h}"
                forecasts.append(
                    (float(row.get(col, 0.0)) - self._price_mean) / self._price_std
                )

        extras = [
            (float(row.get(c, 0.0)) - self._extra_means[c]) / self._extra_stds[c]
            for c in self._extra_cols
        ]

        return np.array(
            [soc_norm, hour_sin, hour_cos, dow_sin, dow_cos, price_z, *forecasts, *extras],
            dtype=np.float32,
        )

    def _validate_columns(self) -> None:
        required = {"price"}
        for q_label in _QUANTILE_LABELS:
            for h in range(1, _HORIZONS + 1):
                col = f"{q_label}_h{h}"
                if col not in self._df.columns:
                    # Missing forecast cols → fill with 0 (raw/no-forecast mode)
                    self._df[col] = 0.0
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(
                f"BatteryEnv: price_df is missing {len(missing)} column(s). "
                f"First few: {sorted(missing)[:5]}"
            )
