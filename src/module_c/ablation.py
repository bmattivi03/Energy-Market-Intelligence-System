"""Module C ablation study: Raw vs B-only vs B+A.

Three observation modes:
  raw   — price only (no forecast columns, zeroed)
  b     — Module B price quantile forecasts (78-dim, current setup)
  ba    — Module B price forecasts + Module A load_q50 h1/6/12/24 (82-dim)

For each mode trains PPO (200k) and SAC (200k + 500k if --full).
Saves checkpoints, produces comparison plots and report.

PYTHONPATH=src python -m module_c.ablation
PYTHONPATH=src python -m module_c.ablation --full   # includes 500k SAC
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import zlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
CKPT_B_DIR   = PROJECT_ROOT / "checkpoints" / "module_b"
CKPT_C_DIR   = PROJECT_ROOT / "checkpoints" / "module_c"
REPORTS_DIR  = PROJECT_ROOT / "reports"
LOAD_Q_PATH  = PROJECT_ROOT / "data" / "module_a" / "load_quantiles.parquet"

PRODUCTION_BUNDLES = ["calendar", "lags", "fundamentals", "spike", "regime", "weather"]
QUANTILES   = (0.1, 0.5, 0.9)
N_EVAL      = 200
SEED        = 42
LOAD_EXTRA  = ["load_q50_h1", "load_q50_h6", "load_q50_h12", "load_q50_h24"]


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_module_b(cqr_dir, catboost_dir):
    from module_b.evaluation import ConformalQuantileRegressor
    from module_b.models import CatBoostQuantileForecaster
    base = CatBoostQuantileForecaster.load(catboost_dir)
    with open(cqr_dir / "cqr_state.pkl", "rb") as f:
        state = pickle.load(f)
    cqr = ConformalQuantileRegressor(base=base, alpha=state["alpha"])
    cqr.delta = state["delta"]
    return cqr


def _make_supervised(full_feat, split_raw, past_cols, fut_cols):
    from module_b.features import prepare_supervised
    X, _ = prepare_supervised(
        full_feat.loc[split_raw.index], past_cols=past_cols, future_cols=fut_cols
    )
    return X


def _build_b_price_df(raw, cqr, X, price_col):
    from module_c.run import _build_price_df
    return _build_price_df(raw, cqr, X, price_col)


def _build_raw_price_df(raw, price_col):
    """price_df with only 'price' — forecast cols zeroed by env."""
    df = raw[[price_col]].copy().rename(columns={price_col: "price"})
    return df.dropna()


def _add_load_features(price_df):
    """Join Module A load_q50 at h1/6/12/24 to an existing price_df."""
    if not LOAD_Q_PATH.exists():
        raise FileNotFoundError(f"load_quantiles.parquet not found: {LOAD_Q_PATH}")
    lq = pd.read_parquet(LOAD_Q_PATH)[LOAD_EXTRA]
    lq.index.name = price_df.index.name
    return price_df.join(lq, how="inner").dropna()


# ── training ──────────────────────────────────────────────────────────────────

def _train_agent(algo, steps, train_df, extra_cols, label, ckpt_path):
    from module_c.environment import BatteryEnv, BatteryEnvConfig
    from module_c.reward import CvarRewardShaper
    from module_c.train import PPOAgent, SACAgent, TrainConfig

    cfg = BatteryEnvConfig(episode_len=24, extra_cols=tuple(extra_cols))
    env = BatteryEnv(train_df, config=cfg,
                     reward_shaper=CvarRewardShaper(lambda_risk=0.1), seed=SEED)

    if (ckpt_path / "model.zip").exists():
        print(f"  [{label}] loading existing checkpoint...")
        cls = PPOAgent if algo == "PPO" else SACAgent
        return cls.load(ckpt_path)

    print(f"  [{label}] training {algo} {steps//1000}k steps...")
    tc = TrainConfig(algo=algo, total_timesteps=steps, lambda_risk=0.1,
                     n_envs=4 if algo == "PPO" else 1, seed=SEED)
    agent = (PPOAgent if algo == "PPO" else SACAgent)()
    agent.train(env, tc)
    agent.save(ckpt_path)
    print(f"  [{label}] saved -> {ckpt_path}")
    return agent


# ── evaluation ────────────────────────────────────────────────────────────────

def _collect(agent, test_df, extra_cols, n, seed):
    from module_c.environment import BatteryEnv, BatteryEnvConfig
    from module_c.reward import CvarRewardShaper
    cfg = BatteryEnvConfig(episode_len=24, extra_cols=tuple(extra_cols))
    env = BatteryEnv(test_df, config=cfg,
                     reward_shaper=CvarRewardShaper(lambda_risk=0.1), seed=seed)
    profits = []
    for _ in range(n):
        obs, _ = env.reset()
        ep = 0.0
        done = False
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            ep += float(info["step_profit"])
            done = term or trunc
        profits.append(ep)
    return np.array(profits)


def _summary(arr):
    thr = np.quantile(arr, 0.05)
    cvar = float(arr[arr <= thr].mean()) if (arr <= thr).any() else float(thr)
    return {
        "mean":   round(float(arr.mean()), 1),
        "std":    round(float(arr.std()), 1),
        "median": round(float(np.median(arr)), 1),
        "cvar":   round(cvar, 1),
        "sharpe": round(float(arr.mean() / arr.std()) if arr.std() > 0 else 0, 3),
        "p10":    round(float(np.quantile(arr, 0.10)), 1),
        "p90":    round(float(np.quantile(arr, 0.90)), 1),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_comparison(all_profits: dict, title: str, path: pathlib.Path):
    """Violin + box plot comparing profit distributions across configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = list(all_profits.keys())
    data   = [all_profits[k] for k in labels]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(labels)))

    # Left: violin
    ax = axes[0]
    parts = ax.violinplot(data, positions=range(len(labels)), showmedians=True, showextrema=False)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    ax.axhline(0, color="red", ls="--", lw=1, alpha=0.6, label="Break-even")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Episode profit (EUR)")
    ax.set_title("Profit distribution (violin)", fontsize=10)
    ax.legend(fontsize=8)

    # Right: mean ± std bar + CVaR markers
    ax2 = axes[1]
    means  = [all_profits[k].mean() for k in labels]
    stds   = [all_profits[k].std()  for k in labels]
    cvars  = [_summary(all_profits[k])["cvar"] for k in labels]
    x = np.arange(len(labels))
    bars = ax2.bar(x, means, yerr=stds, capsize=4,
                   color=colors, edgecolor="white", alpha=0.85)
    ax2.scatter(x, cvars, color="red", zorder=5, s=60, label="CVaR 5%", marker="v")
    ax2.axhline(0, color="black", ls="--", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("EUR per episode")
    ax2.set_title("Mean ± Std  (red ▼ = CVaR 5%)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.bar_label(bars, fmt="%.0f", padding=3, fontsize=7)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Include SAC 500k in ablation (slower)")
    args = parser.parse_args()

    from data.loaders import load_train, load_val, load_test
    from data.schemas import PRICE_COL
    from module_b.features import build_features, prepare_supervised

    REPORTS_DIR.mkdir(exist_ok=True)

    # ── Module B ─────────────────────────────────────────────────────────────
    print("Loading Module B checkpoint...")
    cqr = _load_module_b(CKPT_B_DIR / "cqr", CKPT_B_DIR / "catboost")
    print(f"  CQR delta = {cqr.delta:.4f} EUR/MWh")

    # ── features ─────────────────────────────────────────────────────────────
    print("Building features...")
    train_raw = load_train(); val_raw = load_val(); test_raw = load_test()
    full_feat = build_features(pd.concat([train_raw, val_raw, test_raw]), PRODUCTION_BUNDLES)
    past_cols = [c for c in full_feat.columns if (
        c.startswith("price_lag") or c.startswith("price_rmean") or c.startswith("price_rstd")
        or c in ("residual_load", "renewable_penetration", "clean_spark_anchor",
                 "clean_dark_anchor", "gas_carbon_interaction",
                 "is_high_residual_load", "is_renewable_scarcity", "crisis_2022_flag")
    )]
    fut_cols = [c for c in (
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_weekend", "is_holiday_DE",
        "weather_mean_wind_speed_100m", "weather_mean_shortwave_radiation",
    ) if c in full_feat.columns]

    X_train = _make_supervised(full_feat, train_raw, past_cols, fut_cols)
    X_val   = _make_supervised(full_feat, val_raw,   past_cols, fut_cols)
    X_test  = _make_supervised(full_feat, test_raw,  past_cols, fut_cols)

    # ── build price_dfs for each mode ─────────────────────────────────────────
    print("Building price DataFrames...")
    b_train = _build_b_price_df(train_raw, cqr, X_train, PRICE_COL)
    b_val   = _build_b_price_df(val_raw,   cqr, X_val,   PRICE_COL)
    b_test  = _build_b_price_df(test_raw,  cqr, X_test,  PRICE_COL)
    b_tv    = pd.concat([b_train, b_val]).sort_index()

    raw_train = _build_raw_price_df(train_raw, PRICE_COL)
    raw_val   = _build_raw_price_df(val_raw,   PRICE_COL)
    raw_test  = _build_raw_price_df(test_raw,  PRICE_COL)
    raw_tv    = pd.concat([raw_train, raw_val]).sort_index()

    ba_train = _add_load_features(b_train)
    ba_val   = _add_load_features(b_val)
    ba_test  = _add_load_features(b_test)
    ba_tv    = pd.concat([ba_train, ba_val]).sort_index()

    print(f"  b rows:   tv={len(b_tv):,}  test={len(b_test):,}")
    print(f"  raw rows: tv={len(raw_tv):,} test={len(raw_test):,}")
    print(f"  ba rows:  tv={len(ba_tv):,}  test={len(ba_test):,}")

    # ── training configs ──────────────────────────────────────────────────────
    CKPT_C_DIR.mkdir(parents=True, exist_ok=True)
    sac_steps = [200_000, 500_000] if args.full else [200_000]

    modes = [
        ("raw", raw_tv, raw_test, []),
        ("b",   b_tv,   b_test,  []),
        ("ba",  ba_tv,  ba_test, LOAD_EXTRA),
    ]

    # ── train all ─────────────────────────────────────────────────────────────
    print("\n--- Training ---")
    agents = {}   # (mode, algo, steps) -> agent
    for mode, tv, _, extra in modes:
        # PPO 200k
        tag = f"{mode}_ppo_200k"
        agents[(mode, "PPO", 200_000)] = _train_agent(
            "PPO", 200_000, tv, extra, tag, CKPT_C_DIR / tag
        )
        # SAC
        for steps in sac_steps:
            tag = f"{mode}_sac_{steps//1000}k"
            agents[(mode, "SAC", steps)] = _train_agent(
                "SAC", steps, tv, extra, tag, CKPT_C_DIR / tag
            )

    # ── evaluate all on test ──────────────────────────────────────────────────
    print("\n--- Evaluating on test ---")
    profits = {}   # same keys as agents
    for (mode, algo, steps), agent in agents.items():
        _, _, test_df, extra = next(m for m in modes if m[0] == mode)
        tag = f"{mode}_{algo.lower()}_{steps//1000}k"
        print(f"  {tag}...")
        # Deterministic per-config eval seed (zlib.crc32 is stable across runs,
        # unlike the builtin hash() which varies with PYTHONHASHSEED).
        eval_seed = SEED + zlib.crc32(tag.encode()) % 100
        profits[(mode, algo, steps)] = _collect(agent, test_df, extra, N_EVAL, eval_seed)

    # ── print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Config':<22} {'Mean':>8} {'Std':>7} {'Median':>8} {'CVaR5%':>8} {'Sharpe':>7}")
    print(f"{'='*80}")
    for (mode, algo, steps), arr in profits.items():
        s = _summary(arr)
        label = f"{mode} | {algo} {steps//1000}k"
        print(f"{label:<22} {s['mean']:>8.0f} {s['std']:>7.0f} {s['median']:>8.0f} "
              f"{s['cvar']:>8.0f} {s['sharpe']:>7.3f}")
    print(f"{'='*80}")

    # ── ablation comparison plots ─────────────────────────────────────────────
    print("\n--- Generating plots ---")
    best_sac_steps = max(sac_steps)

    # 1. Main A→C ablation: raw vs B vs B+A for the best SAC config.
    ablation_profits = {
        "Raw\n(no forecast)": profits[("raw", "SAC", best_sac_steps)],
        "B only\n(price Q)":  profits[("b",   "SAC", best_sac_steps)],
        "B + A\n(price+load)":profits[("ba",  "SAC", best_sac_steps)],
    }
    _plot_comparison(
        ablation_profits,
        f"Module C ablation: observation mode — SAC {best_sac_steps//1000}k (2025 Q1 test)",
        REPORTS_DIR / "module_c_ablation_sac.png",
    )

    # 2. Full grid: all modes × both algos (best steps).
    full_profits = {}
    for mode in ["raw", "b", "ba"]:
        for algo in ["PPO", "SAC"]:
            steps = best_sac_steps if algo == "SAC" else 200_000
            key = f"{mode.upper()}\n{algo} {steps//1000}k"
            full_profits[key] = profits[(mode, algo, steps)]
    _plot_comparison(
        full_profits,
        "Module C: full ablation grid (2025 Q1 test, 100MWh/50MW battery)",
        REPORTS_DIR / "module_c_ablation_full.png",
    )

    # ── decide whether A→C helps and record it ────────────────────────────────
    def _mean(mode, algo, steps):
        return float(profits[(mode, algo, steps)].mean())

    b_mean  = _mean("b",  "SAC", best_sac_steps)
    ba_mean = _mean("ba", "SAC", best_sac_steps)
    a_helps = ba_mean > b_mean
    verdict = (
        f"B+A mean profit {ba_mean:.0f} EUR vs B-only {b_mean:.0f} EUR "
        f"({'+' if a_helps else ''}{ba_mean - b_mean:.0f}). "
        + ("Module A load quantiles HELP — keep A→C wired (extra_cols)."
           if a_helps else
           "Module A load quantiles do NOT help — leave C on B-only (default).")
    )
    print("\n" + verdict)

    # ── save JSON ─────────────────────────────────────────────────────────────
    result_json = {
        k[0] + "|" + k[1] + "|" + str(k[2]): _summary(v)
        for k, v in profits.items()
    }
    (REPORTS_DIR / "module_c_ablation_results.json").write_text(
        json.dumps({
            "n_eval": N_EVAL,
            "load_extra_cols": LOAD_EXTRA,
            "a_to_c_helps": a_helps,
            "verdict": verdict,
            "results": result_json,
        }, indent=2)
    )
    print(f"\nSaved: reports/module_c_ablation_results.json")


if __name__ == "__main__":
    main()
