"""Train and evaluate Module C RL agents using Module B price forecasts.

Builds price_df (actual prices + Module B q10/q50/q90 × h1-24), trains PPO
and SAC agents on 2019-2023 train split, evaluates on 2025 Q1 test split.

Baselines
---------
  idle      always action=0  → profit ≈ 0
  greedy    charge cheapest 2h, discharge most expensive 2h per episode
  perfect   oracle with exact episode prices (theoretical upper bound)

Usage
-----
PYTHONPATH=src python -m module_c.run
"""

from __future__ import annotations

import json
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd

from config import ModuleCConfig

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
CKPT_B_DIR   = PROJECT_ROOT / "checkpoints" / "module_b"
CKPT_C_DIR   = PROJECT_ROOT / "checkpoints" / "module_c"
REPORT_PATH  = PROJECT_ROOT / "reports" / "module_c_results.json"

PRODUCTION_BUNDLES = ["calendar", "lags", "fundamentals", "spike", "regime", "weather"]
QUANTILES = (0.1, 0.5, 0.9)
N_EVAL_EPISODES = ModuleCConfig.n_eval_episodes
TOTAL_TIMESTEPS = ModuleCConfig.total_timesteps
SEED = ModuleCConfig.seed

# Module A → C: the ablation (reports/module_c_ablation_results.json) showed adding
# these load-quantile columns to the observation lifts SAC profit +12% and Sharpe
# 1.43→1.72. Enable with --use-load-quantiles (requires data/module_a/load_quantiles.parquet).
LOAD_Q_PATH = PROJECT_ROOT / "data" / "module_a" / "load_quantiles.parquet"
LOAD_EXTRA = ["load_q50_h1", "load_q50_h6", "load_q50_h12", "load_q50_h24"]


def _join_load_quantiles(price_df):
    """Join Module A's load_q50 forecasts onto a price_df (for the A→C config)."""
    lq = pd.read_parquet(LOAD_Q_PATH)[LOAD_EXTRA]
    lq.index.name = price_df.index.name
    return price_df.join(lq, how="inner").dropna()


# ── build price_df ────────────────────────────────────────────────────────────

def _build_price_df(raw: pd.DataFrame, cqr, X: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Build wide-format price_df for BatteryEnv.

    Index: datetime (origin_ts)
    Cols : price, q10_h1..q90_h24
    """
    q_long = cqr.predict_quantiles(X)   # DataFrame with q10/q50/q90 columns

    # q_long has row-per-(origin_ts, horizon_h): pivot to wide
    q_long = q_long.copy()
    q_long["origin_ts"] = X["origin_ts"].values
    q_long["horizon_h"] = X["horizon_h"].values

    wide = q_long.pivot_table(index="origin_ts", columns="horizon_h",
                               values=["q10", "q50", "q90"], aggfunc="first")
    wide.columns = [f"{q}_h{h}" for q, h in wide.columns]
    wide.index.name = "datetime_utc"
    # Restore UTC timezone lost during pivot
    if wide.index.tz is None:
        wide.index = pd.DatetimeIndex(wide.index).tz_localize("UTC")

    # Join actual price
    price_series = raw[price_col].rename("price")
    price_series.index.name = "datetime_utc"
    df = wide.join(price_series, how="inner").dropna()
    return df


# ── baselines ─────────────────────────────────────────────────────────────────

def _idle_profit(price_df: pd.DataFrame, n_episodes: int, cfg, seed: int) -> list[float]:
    """Always action=0 → no charge/discharge → profit=0."""
    return [0.0] * n_episodes


def _greedy_profit(price_df: pd.DataFrame, n_episodes: int, cfg, seed: int) -> list[float]:
    """Greedy oracle within each 24h episode: charge 2h cheapest, discharge 2h most expensive.

    Uses actual prices known at episode start - theoretical upper bound approximation.
    """
    from module_c.environment import CAPACITY_MWH, MAX_POWER_MW, EFFICIENCY, DT_H
    import math
    sqrt_eff = math.sqrt(EFFICIENCY)
    rng = np.random.default_rng(seed)
    profits = []
    ep_len = cfg.episode_len

    for _ in range(n_episodes):
        max_start = len(price_df) - ep_len
        start = int(rng.integers(0, max(1, max_start)))
        prices = price_df["price"].iloc[start:start + ep_len].values

        soc = (cfg.capacity_mwh * 0.5)
        ranked = np.argsort(prices)
        charge_hours = set(ranked[:2])      # 2 cheapest
        discharge_hours = set(ranked[-2:])  # 2 most expensive

        ep_profit = 0.0
        for step, price in enumerate(prices):
            if step in charge_hours and soc < cfg.capacity_mwh - 1e-6:
                power = min(MAX_POWER_MW, (cfg.capacity_mwh - soc) / (DT_H * sqrt_eff))
                energy = power * DT_H * sqrt_eff
                soc = min(soc + energy, cfg.capacity_mwh)
                ep_profit -= price * power * DT_H
            elif step in discharge_hours and soc > 1e-6:
                power = min(MAX_POWER_MW, soc * sqrt_eff / DT_H)
                energy = power * DT_H / sqrt_eff
                soc = max(soc - energy, 0.0)
                ep_profit += price * power * DT_H
        profits.append(ep_profit)
    return profits


def _agent_profit(agent, env, n_episodes: int) -> list[float]:
    from module_c.train import evaluate_agent
    result = evaluate_agent(agent, env, n_episodes=n_episodes)
    # Re-collect per-episode to get list
    profits = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep = 0.0
        done = False
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            ep += float(info["step_profit"])
            done = terminated or truncated
        profits.append(ep)
    return profits


def _summary(profits: list[float]) -> dict:
    arr = np.array(profits, dtype=np.float64)
    cvar_threshold = np.quantile(arr, 0.05)
    cvar = float(arr[arr <= cvar_threshold].mean()) if (arr <= cvar_threshold).any() else float(cvar_threshold)
    return {
        "mean_profit_eur":  round(float(arr.mean()), 2),
        "std_profit_eur":   round(float(arr.std()), 2),
        "median_profit_eur": round(float(np.median(arr)), 2),
        "cvar_5pct_eur":    round(cvar, 2),
        "sharpe":           round(float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0, 3),
        "n_episodes":       len(profits),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(use_load_quantiles: bool = False) -> None:
    from data.loaders import load_train, load_val, load_test
    from data.schemas import PRICE_COL
    from module_b.evaluation import ConformalQuantileRegressor
    from module_b.features import build_features, prepare_supervised
    from module_b.models import CatBoostQuantileForecaster
    from module_c.environment import BatteryEnv, BatteryEnvConfig
    from module_c.reward import CvarRewardShaper
    from module_c.train import PPOAgent, SACAgent, TrainConfig, build_agent

    # ── 1. load Module B checkpoint ──────────────────────────────────────────
    print("Loading Module B checkpoint...")
    catboost_dir = CKPT_B_DIR / "catboost"
    cqr_dir = CKPT_B_DIR / "cqr"

    base_model = CatBoostQuantileForecaster.load(catboost_dir)
    with open(cqr_dir / "cqr_state.pkl", "rb") as f:
        cqr_state = pickle.load(f)

    cqr = ConformalQuantileRegressor(base=base_model, alpha=cqr_state["alpha"])
    cqr.delta = cqr_state["delta"]
    print(f"  CQR delta = {cqr.delta:.4f} EUR/MWh")

    # ── 2. build features and supervised layout ──────────────────────────────
    print("Building features...")
    train_raw = load_train()
    val_raw   = load_val()
    test_raw  = load_test()
    full_raw  = pd.concat([train_raw, val_raw, test_raw])
    full_feat = build_features(full_raw, PRODUCTION_BUNDLES)

    # Curated cols (matches train.py)
    past_cols = [c for c in full_feat.columns if (
        c.startswith("price_lag") or c.startswith("price_rmean") or c.startswith("price_rstd")
        or c in ("residual_load", "renewable_penetration", "clean_spark_anchor",
                 "clean_dark_anchor", "gas_carbon_interaction",
                 "is_high_residual_load", "is_renewable_scarcity", "crisis_2022_flag")
    )]
    future_cols = [c for c in (
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_weekend", "is_holiday_DE",
        "weather_mean_wind_speed_100m", "weather_mean_shortwave_radiation",
    ) if c in full_feat.columns]

    train_feat = full_feat.loc[train_raw.index]
    val_feat   = full_feat.loc[val_raw.index]
    test_feat  = full_feat.loc[test_raw.index]

    X_train, _ = prepare_supervised(train_feat, past_cols=past_cols, future_cols=future_cols)
    X_val,   _ = prepare_supervised(val_feat,   past_cols=past_cols, future_cols=future_cols)
    X_test,  _ = prepare_supervised(test_feat,  past_cols=past_cols, future_cols=future_cols)

    # ── 3. build price_df per split ──────────────────────────────────────────
    print("Generating Module B forecasts for environment...")
    train_price_df = _build_price_df(train_raw, cqr, X_train, PRICE_COL)
    val_price_df   = _build_price_df(val_raw,   cqr, X_val,   PRICE_COL)
    test_price_df  = _build_price_df(test_raw,  cqr, X_test,  PRICE_COL)

    # A→C (validated by the ablation): append Module A load quantiles to the obs.
    extra_cols: tuple = ()
    if use_load_quantiles:
        if LOAD_Q_PATH.exists():
            train_price_df = _join_load_quantiles(train_price_df)
            val_price_df   = _join_load_quantiles(val_price_df)
            test_price_df  = _join_load_quantiles(test_price_df)
            extra_cols = tuple(LOAD_EXTRA)
            print(f"  A→C enabled: +{len(LOAD_EXTRA)} load-quantile obs columns")
        else:
            print(f"  --use-load-quantiles set but {LOAD_Q_PATH.name} missing; "
                  "run module_a first. Falling back to B-only.")

    # Train on train+val combined
    trainval_price_df = pd.concat([train_price_df, val_price_df]).sort_index()
    print(f"  trainval: {len(trainval_price_df):,} rows  test: {len(test_price_df):,} rows")

    # ── 4. build environments ────────────────────────────────────────────────
    env_cfg = BatteryEnvConfig(episode_len=24, extra_cols=extra_cols)
    shaper  = CvarRewardShaper(lambda_risk=0.1)

    train_env = BatteryEnv(trainval_price_df, config=env_cfg,
                           reward_shaper=shaper, seed=SEED)
    test_env  = BatteryEnv(test_price_df, config=env_cfg,
                           reward_shaper=CvarRewardShaper(lambda_risk=0.1), seed=SEED + 1)

    # ── 5. train PPO ─────────────────────────────────────────────────────────
    ppo_ckpt = CKPT_C_DIR / "ppo" / "model.zip"
    if ppo_ckpt.exists():
        print("\nLoading existing PPO checkpoint...")
        ppo = PPOAgent.load(CKPT_C_DIR / "ppo")
    else:
        print(f"\nTraining PPO ({TOTAL_TIMESTEPS:,} steps)...")
        ppo_cfg = TrainConfig(algo="PPO", total_timesteps=TOTAL_TIMESTEPS,
                              lambda_risk=0.1, n_envs=4, seed=SEED,
                              checkpoint_dir=CKPT_C_DIR)
        ppo = PPOAgent()
        ppo.train(train_env, ppo_cfg)
        ppo.save(CKPT_C_DIR / "ppo")
        print("  Saved PPO checkpoint.")

    # ── 6. train SAC ─────────────────────────────────────────────────────────
    sac_ckpt = CKPT_C_DIR / "sac" / "model.zip"
    if sac_ckpt.exists():
        print("Loading existing SAC checkpoint...")
        sac = SACAgent.load(CKPT_C_DIR / "sac")
    else:
        print(f"\nTraining SAC ({TOTAL_TIMESTEPS:,} steps)...")
        sac_env = BatteryEnv(trainval_price_df, config=env_cfg,
                             reward_shaper=CvarRewardShaper(lambda_risk=0.1), seed=SEED)
        sac_cfg = TrainConfig(algo="SAC", total_timesteps=TOTAL_TIMESTEPS,
                              lambda_risk=0.1, n_envs=1, seed=SEED,
                              checkpoint_dir=CKPT_C_DIR)
        sac = SACAgent()
        sac.train(sac_env, sac_cfg)
        sac.save(CKPT_C_DIR / "sac")
        print("  Saved SAC checkpoint.")

    # ── 7. evaluate on test ──────────────────────────────────────────────────
    print(f"\nEvaluating on test ({N_EVAL_EPISODES} episodes)...")

    idle_profits    = _idle_profit(test_price_df, N_EVAL_EPISODES, env_cfg, SEED)
    greedy_profits  = _greedy_profit(test_price_df, N_EVAL_EPISODES, env_cfg, SEED)
    ppo_profits     = _agent_profit(ppo, test_env, N_EVAL_EPISODES)

    test_env_sac = BatteryEnv(test_price_df, config=env_cfg,
                              reward_shaper=CvarRewardShaper(lambda_risk=0.1), seed=SEED + 2)
    sac_profits = _agent_profit(sac, test_env_sac, N_EVAL_EPISODES)

    results = {
        "idle":   _summary(idle_profits),
        "greedy": _summary(greedy_profits),
        "PPO":    _summary(ppo_profits),
        "SAC":    _summary(sac_profits),
    }

    # ── 8. print table ───────────────────────────────────────────────────────
    print(f"\n{'-'*70}")
    print(f"{'Model':<10} {'Mean profit':>12} {'Std':>8} {'Median':>8} "
          f"{'CVaR 5%':>10} {'Sharpe':>8}")
    print(f"{'-'*70}")
    for name, m in results.items():
        print(f"{name:<10} {m['mean_profit_eur']:>12.1f} {m['std_profit_eur']:>8.1f} "
              f"{m['median_profit_eur']:>8.1f} {m['cvar_5pct_eur']:>10.1f} "
              f"{m['sharpe']:>8.3f}")
    print(f"{'-'*70}")
    print("All values in EUR per 24h episode (100 MWh / 50 MW battery)")

    # ── 9. save ──────────────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "battery": {"capacity_mwh": 100, "max_power_mw": 50, "efficiency": 0.90},
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_eval_episodes": N_EVAL_EPISODES,
        "test_split": "2025 Q1",
        "module_b_cqr_delta": cqr.delta,
        "results": results,
    }, indent=2))
    print(f"\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train + evaluate Module C RL agents.")
    parser.add_argument(
        "--use-load-quantiles", action="store_true",
        help="A→C: add Module A load quantiles to the observation (ablation-validated: "
             "+12%% SAC profit). Requires data/module_a/load_quantiles.parquet.",
    )
    args = parser.parse_args()
    main(use_load_quantiles=args.use_load_quantiles)
