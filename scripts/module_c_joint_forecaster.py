"""Does the joint-DL forecaster help the RL (B->C uncertainty propagation)?

Trains the joint-DL price model (v2 winning config) on 2019-2023, builds the same
wide price_df Module C expects (price + q10/q50/q90_h1..h24), trains FRESH PPO+SAC
on it (seed 42, 200k steps, separate checkpoint dir), evaluates on 2025-Q1, and
compares to the existing CatBoost-fed Module C result. The actual `price` column
is identical to the CatBoost run, so only the forecast columns differ.

Usage:
  PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      python scripts/module_c_joint_forecaster.py
  (add --sanity for a fast plumbing check)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch  # noqa: E402  before module_b / sb3
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
from data.loaders import load_train, load_val, load_test  # noqa: E402
from data.schemas import PRICE_COL  # noqa: E402
from module_joint.config import JointConfig, select_device  # noqa: E402
from module_joint.ensemble import train_ensemble  # noqa: E402

PROJECT_ROOT = Path(".")
CKPT_C_JOINT = PROJECT_ROOT / "checkpoints" / "module_c_joint"
REPORT = PROJECT_ROOT / "reports" / "module_c_joint_results.json"
LQ_PATH = "data/module_a/load_quantiles.parquet"
LOOKBACK = 168


def joint_wide_price_df(ens, split_raw, ctx_prev, lq):
    """Wide price_df from the joint ensemble for one split: price + q*_h1..h24."""
    ctx = split_raw if ctx_prev is None else pd.concat([ctx_prev.tail(LOOKBACK), split_raw])
    long = ens.predict_quantiles(ctx, lq, restrict_to=split_raw.index)["price"]
    w = long.reset_index().pivot_table(index="origin_ts", columns="horizon_h",
                                       values=["q10", "q50", "q90"], aggfunc="first")
    w.columns = [f"{q}_h{h}" for q, h in w.columns]
    w.index.name = "datetime_utc"
    if w.index.tz is None:
        w.index = pd.DatetimeIndex(w.index).tz_localize("UTC")
    price = split_raw[PRICE_COL].rename("price")
    price.index.name = "datetime_utc"
    return w.join(price, how="inner").dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--rl-seed", type=int, default=42)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    if args.sanity:
        args.seeds, args.epochs, args.timesteps, args.episodes = "0", 2, 2000, 10

    device = select_device()
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[c-joint] device={device} jseeds={seeds} timesteps={args.timesteps}", flush=True)

    lq = pd.read_parquet(LQ_PATH)
    train, val, test = load_train(), load_val(), load_test()

    # 1) joint-DL ensemble (v2 winning config) on 2019-2023
    cfg = JointConfig(backbone="tide", use_load_quantiles=True, load_to_price=False,
                      fundamentals=True, aux_residual_load=False, aux_spike=False,
                      aux_renewables=False, max_epochs=args.epochs, patience=18)
    t0 = time.time()
    ens = train_ensemble(cfg, seeds, train, val, lq, device=device)
    print(f"[c-joint] joint ensemble trained in {time.time()-t0:.0f}s", flush=True)

    # 2) wide price_dfs
    train_pdf = joint_wide_price_df(ens, train, None, lq)
    val_pdf = joint_wide_price_df(ens, val, train, lq)
    test_pdf = joint_wide_price_df(ens, test, val, lq)
    trainval_pdf = pd.concat([train_pdf, val_pdf]).sort_index()
    print(f"[c-joint] price_df trainval={len(trainval_pdf):,} test={len(test_pdf):,}", flush=True)

    # 3) RL: fresh PPO + SAC on the joint forecasts (mirror run.py protocol)
    from module_c.environment import BatteryEnv, BatteryEnvConfig
    from module_c.reward import CvarRewardShaper
    from module_c.train import PPOAgent, SACAgent, TrainConfig
    from module_c.run import _idle_profit, _greedy_profit, _agent_profit, _summary

    env_cfg = BatteryEnvConfig(episode_len=24)
    S = args.rl_seed
    train_env = BatteryEnv(trainval_pdf, config=env_cfg,
                           reward_shaper=CvarRewardShaper(lambda_risk=0.1), seed=S)
    test_env = BatteryEnv(test_pdf, config=env_cfg,
                          reward_shaper=CvarRewardShaper(lambda_risk=0.1), seed=S + 1)

    t0 = time.time()
    ppo = PPOAgent()
    ppo.train(train_env, TrainConfig(algo="PPO", total_timesteps=args.timesteps,
                                     lambda_risk=0.1, n_envs=4, seed=S, checkpoint_dir=CKPT_C_JOINT))
    ppo.save(CKPT_C_JOINT / "ppo")
    sac_env = BatteryEnv(trainval_pdf, config=env_cfg,
                         reward_shaper=CvarRewardShaper(lambda_risk=0.1), seed=S)
    sac = SACAgent()
    sac.train(sac_env, TrainConfig(algo="SAC", total_timesteps=args.timesteps,
                                   lambda_risk=0.1, n_envs=1, seed=S, checkpoint_dir=CKPT_C_JOINT))
    sac.save(CKPT_C_JOINT / "sac")
    print(f"[c-joint] RL trained in {time.time()-t0:.0f}s", flush=True)

    # 4) evaluate
    idle = _idle_profit(test_pdf, args.episodes, env_cfg, S)
    greedy = _greedy_profit(test_pdf, args.episodes, env_cfg, S)
    ppo_p = _agent_profit(ppo, test_env, args.episodes)
    test_env_sac = BatteryEnv(test_pdf, config=env_cfg,
                              reward_shaper=CvarRewardShaper(lambda_risk=0.1), seed=S + 2)
    sac_p = _agent_profit(sac, test_env_sac, args.episodes)
    results = {"idle": _summary(idle), "greedy": _summary(greedy),
               "PPO": _summary(ppo_p), "SAC": _summary(sac_p)}

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"forecaster": "joint_dl_2019_2023",
                                  "test_split": "2025 Q1", "results": results}, indent=2))

    # 5) compare to CatBoost-fed Module C
    cb = json.load(open("reports/module_c_results.json"))["results"]
    print("\n[c-joint] ===== RL with JOINT-DL vs CatBoost forecasts (2025-Q1 test) =====", flush=True)
    print(f"{'model':<8}{'metric':<16}{'JOINT-DL':>12}{'CatBoost':>12}{'delta':>12}", flush=True)
    for mdl in ("PPO", "SAC", "greedy"):
        for met in ("mean_profit_eur", "sharpe", "cvar_5pct_eur"):
            j, c = results[mdl][met], cb[mdl][met]
            print(f"{mdl:<8}{met:<16}{j:>12.2f}{c:>12.2f}{j-c:>+12.2f}", flush=True)
    # capture ratio (agent / greedy-oracle) normalizes for the different test pools
    print("\n[c-joint] capture ratio = agent_mean_profit / greedy_mean_profit "
          "(controls for episode-pool differences):", flush=True)
    jg, cg = results["greedy"]["mean_profit_eur"], cb["greedy"]["mean_profit_eur"]
    print(f"{'model':<8}{'JOINT-DL':>12}{'CatBoost':>12}{'delta':>12}", flush=True)
    for mdl in ("PPO", "SAC"):
        jr = results[mdl]["mean_profit_eur"] / jg if jg else float("nan")
        cr = cb[mdl]["mean_profit_eur"] / cg if cg else float("nan")
        print(f"{mdl:<8}{jr:>12.3f}{cr:>12.3f}{jr-cr:>+12.3f}", flush=True)
    print(f"\n[c-joint] joint greedy={jg:.0f} vs catboost greedy={cg:.0f} "
          f"(pool opportunity); n_test joint={len(test_pdf)}", flush=True)
    print(f"[c-joint] saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
