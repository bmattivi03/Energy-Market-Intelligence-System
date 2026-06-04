#!/usr/bin/env python3
"""Generate every regenerated figure for the Energy Market Intelligence System deck.

Reads committed report data (parquet / JSON) and writes clean, palette-consistent PNGs
into ``presentation/figures/``.  matplotlib-only (no seaborn).  Run from anywhere:

    python3 presentation/figures.py

The source ``*.parquet`` / ``data/*`` files are git-ignored, so this script must be run
*locally* (after the pipeline has produced the splits); the resulting PNGs are committed
and are all that Overleaf ever sees.  Each generator is wrapped in try/except so a missing
input never aborts the whole run.

NUMBER HYGIENE: all numbers come from the current SSHB-split artifacts
(``reports/phase7_rerun_summary.md`` is the source of truth).  Do NOT "correct" any
hard-coded value to a stale figure found elsewhere.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Shared palette + matplotlib styling (mirrors the Beamer preamble exactly)
# --------------------------------------------------------------------------------------
EMIS = {
    "ink": "#1A2331",
    "muted": "#5B6573",
    "grid": "#D9DEE5",
    "accent": "#0E7C86",   # data / global teal
    "imp": "#7A5195",      # imputation / SSHB purple
    "A": "#2C7FB8",        # Module A blue
    "B": "#D9772B",        # Module B amber
    "C": "#2E8B57",        # Module C green
    "good": "#2E8B57",
    "warn": "#E0A800",
    "bad": "#C0392B",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.edgecolor": EMIS["muted"],
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": EMIS["grid"],
    "grid.linewidth": 0.7,
    "text.color": EMIS["ink"],
    "axes.labelcolor": EMIS["ink"],
    "xtick.color": EMIS["muted"],
    "ytick.color": EMIS["muted"],
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def _annotate_bars(ax, bars, fmt="{:.0f}", dy=0.0, color=None, fontsize=11, fontweight="bold"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h + dy),
                    ha="center", va="bottom", fontsize=fontsize, fontweight=fontweight,
                    color=color or EMIS["ink"])


# ======================================================================================
# 1. Raw price volatility (slide 2) — motivates "predict bands, not points"
# ======================================================================================
def fig_raw_price():
    df = pd.read_parquet(ROOT / "data/splits/test.parquet")
    s = df["price"].dropna()
    hi = s.quantile(0.95)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(s.index, s.values, color=EMIS["B"], lw=0.9)
    ax.axhline(0, color=EMIS["muted"], lw=0.8, ls="--")
    # mark spikes and negative prices
    spikes = s[s > hi]
    negs = s[s < 0]
    ax.scatter(spikes.index, spikes.values, s=10, color=EMIS["bad"], zorder=3,
               label=f"price > 95th pct ({hi:.0f} €/MWh)")
    if len(negs):
        ax.scatter(negs.index, negs.values, s=10, color=EMIS["accent"], zorder=3,
                   label="negative prices")
    ax.set_ylabel("Day-ahead price  (€/MWh)")
    ax.set_title("DE–LU day-ahead price — locked test window (2025 Q1)")
    ax.annotate(f"max {s.max():.0f}", (s.idxmax(), s.max()), fontsize=10,
                color=EMIS["bad"], xytext=(6, -2), textcoords="offset points")
    ax.annotate(f"min {s.min():.0f}", (s.idxmin(), s.min()), fontsize=10,
                color=EMIS["accent"], xytext=(6, 2), textcoords="offset points")
    ax.legend(loc="upper right", fontsize=10)
    fig.autofmt_xdate()
    _save(fig, "raw_price_volatility.png")


# ======================================================================================
# 2. Missingness heatmap (slide 6) — structured, not random
# ======================================================================================
def fig_missingness():
    mask = pd.read_parquet(ROOT / "data/processed/emis_mask.parquet")
    # 1 = missing/imputed.  Resample to weekly mean -> fraction missing per column per week.
    weekly = mask.astype(float).resample("W").mean()
    cols = list(mask.columns)
    M = weekly[cols].to_numpy().T            # rows = columns, cols = weeks
    n_rows, n_weeks = M.shape
    idx = weekly.index
    cmap = LinearSegmentedColormap.from_list("emis_miss", ["#FFFFFF", EMIS["bad"]])

    def wk(date):  # week column index for a date
        return int(idx.searchsorted(pd.Timestamp(date, tz="UTC")))

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   extent=[0, n_weeks, n_rows, 0], interpolation="nearest")
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(cols, fontsize=6.5)
    # year ticks
    years = list(range(2019, 2026))
    xticks = [wk(f"{y}-01-01") for y in years]
    ax.set_xticks(xticks)
    ax.set_xticklabels(years)
    ax.set_xlabel("time (weekly bins)")
    ax.set_title("Raw-data missingness (47 variables × time) — red = missing / imputed")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("fraction missing", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    def _row(name):
        return cols.index(name) + 0.5 if name in cols else None

    r_cg = _row("gen_fossil_coal_gas")
    if r_cg is not None:
        ax.annotate("coal-gas: no ENTSO-E data 2019–2022",
                    xy=(wk("2020-09-01"), r_cg), xytext=(wk("2021-01-01"), r_cg - 6.5),
                    fontsize=9, color=EMIS["ink"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=EMIS["ink"], lw=1.1))
    r_nuc = _row("gen_nuclear")
    if r_nuc is not None:
        ax.annotate("nuclear → 0 after 2023-04-15 shutdown",
                    xy=(wk("2023-05-01"), r_nuc), xytext=(wk("2019-08-01"), r_nuc + 7),
                    fontsize=9, color=EMIS["ink"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=EMIS["ink"], lw=1.1))
    _save(fig, "missingness_heatmap.png")


# ======================================================================================
# 3. SSHB vs same-model-baseline audit status (slide 8)
# ======================================================================================
def fig_sshb_audit():
    base = json.load(open(ROOT / "reports/imputation_audit_sshbrun_baseline.json"))
    sshb = json.load(open(ROOT / "reports/imputation_audit_sshb.json"))
    cb = Counter(x["status"] for x in base)
    cs = Counter(x["status"] for x in sshb)
    cats = ["green", "yellow", "red"]
    labels = ["green\n(good)", "yellow\n(usable)", "red\n(structural)"]
    colors = [EMIS["good"], EMIS["warn"], EMIS["bad"]]
    base_v = [cb.get(c, 0) for c in cats]
    sshb_v = [cs.get(c, 0) for c in cats]

    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    b1 = ax.bar(x - w / 2, base_v, w, color=colors, alpha=0.45, hatch="//",
                edgecolor="white", linewidth=0)
    b2 = ax.bar(x + w / 2, sshb_v, w, color=colors, edgecolor="white", linewidth=0)
    _annotate_bars(ax, b1, fmt="{:.0f}")
    _annotate_bars(ax, b2, fmt="{:.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("number of variables (of 24)")
    ax.set_ylim(0, max(base_v + sshb_v) + 2)
    ax.set_title("Imputation quality — same model, SSHB inference OFF → ON")
    legend = [Patch(facecolor=EMIS["muted"], alpha=0.45, hatch="//", label="baseline (SSHB off)"),
              Patch(facecolor=EMIS["muted"], label="SSHB on")]
    ax.legend(handles=legend, loc="upper left", fontsize=11)
    ax.annotate("+3 promoted to green", xy=(0 + w / 2, sshb_v[0]),
                xytext=(0.25, sshb_v[0] + 1.2), fontsize=11, color=EMIS["good"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=EMIS["good"], lw=1.2))
    _save(fig, "sshb_audit_status.png")


# ======================================================================================
# 4. Module B per-quantile calibration fix (slide 12) — hard-coded canonical (2024 val)
# ======================================================================================
def fig_module_b_calib_fix():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))

    # left: % of actuals above q90 (ideal = 10%)
    vals = [35.5, 6.8]
    labels = ["single\nMultiQuantile", "per-quantile\n(production)"]
    bars = ax1.bar(labels, vals, color=[EMIS["bad"], EMIS["good"]], width=0.6,
                   edgecolor="white")
    ax1.axhline(10, color=EMIS["muted"], ls="--", lw=1.2)
    ax1.annotate("ideal 10%", (1.35, 10), fontsize=10, color=EMIS["muted"],
                 va="bottom", ha="right")
    _annotate_bars(ax1, bars, fmt="{:.1f}%")
    ax1.set_ylabel("actuals above q90  (%)")
    ax1.set_ylim(0, 42)
    ax1.set_title("Upper-tail calibration")

    # right: relative CQR interval-delta reduction
    bars2 = ax2.bar(["before", "after"], [100, 46],
                    color=[EMIS["bad"], EMIS["good"]], width=0.6, edgecolor="white")
    _annotate_bars(ax2, bars2, fmt="{:.0f}")
    ax2.set_ylabel("CQR interval correction  (index)")
    ax2.set_ylim(0, 118)
    ax2.set_title("CQR widening needed  (−54%)")
    fig.suptitle("Diagnosing & fixing our own miscalibration (2024 validation)",
                 fontsize=14, fontweight="bold", y=1.02)
    _save(fig, "module_b_calibration_fix.png")


# ======================================================================================
# 5. Module B leaderboard (slide 13) — current parquet (canonical 23.29)
# ======================================================================================
def fig_module_b_leaderboard():
    df = pd.read_parquet(ROOT / "reports/module_b_final_leaderboard.parquet")
    d = df[df["horizon_group"] == "h1_6"].set_index("model")
    order = ["naive", "sn168", "lightgbm+cqr", "catboost+cqr"]
    nice = {"naive": "Naive\n(t−24h)", "sn168": "Seasonal\nnaive (168h)",
            "lightgbm+cqr": "LightGBM\n+CQR", "catboost+cqr": "CatBoost+CQR\n(production)"}
    d = d.loc[order]
    colors = [EMIS["muted"], EMIS["muted"], EMIS["A"], EMIS["B"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    xs = np.arange(len(order))
    b1 = ax1.bar(xs, d["mae"].values, color=colors, width=0.62, edgecolor="white")
    _annotate_bars(ax1, b1, fmt="{:.1f}")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([nice[m] for m in order], fontsize=10)
    ax1.set_ylabel("MAE  (€/MWh)")
    ax1.set_title("Point accuracy (h1–6)")
    ax1.set_ylim(0, max(d["mae"]) * 1.18)

    b2 = ax2.bar(xs, d["pinball_avg"].values, color=colors, width=0.62, edgecolor="white")
    _annotate_bars(ax2, b2, fmt="{:.1f}")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([nice[m] for m in order], fontsize=10)
    ax2.set_ylabel("avg pinball loss")
    ax2.set_title("Probabilistic accuracy (h1–6)")
    ax2.set_ylim(0, max(d["pinball_avg"]) * 1.18)

    cb_mae = d.loc["catboost+cqr", "mae"]
    sn_mae = d.loc["sn168", "mae"]
    fig.suptitle(f"Production model: {cb_mae:.2f} €/MWh — "
                 f"{100*(sn_mae-cb_mae)/sn_mae:.0f}% below seasonal-naive (DM p<0.001)",
                 fontsize=13.5, fontweight="bold", y=1.02)
    _save(fig, "module_b_leaderboard.png")


# ======================================================================================
# 6. Module C profit comparison (slide 16) — SAC beats the oracle
# ======================================================================================
def fig_module_c_profit():
    d = json.load(open(ROOT / "reports/module_c_results.json"))["results"]
    order = ["idle", "greedy", "PPO", "SAC"]
    nice = {"idle": "Idle", "greedy": "Greedy\n(perfect-foresight\noracle)",
            "PPO": "PPO", "SAC": "SAC\n(ours)"}
    colors = {"idle": EMIS["grid"], "greedy": EMIS["muted"], "PPO": EMIS["A"], "SAC": EMIS["C"]}
    means = [d[k]["mean_profit_eur"] for k in order]
    stds = [d[k]["std_profit_eur"] for k in order]
    cvars = [d[k]["cvar_5pct_eur"] for k in order]

    fig, ax = plt.subplots(figsize=(9, 5.0))
    xs = np.arange(len(order))
    bars = ax.bar(xs, means, color=[colors[k] for k in order], width=0.6,
                  yerr=stds, capsize=5, edgecolor="white",
                  error_kw=dict(ecolor=EMIS["muted"], lw=1.2))
    for i, k in enumerate(order):
        ax.annotate(f"€{means[i]:,.0f}", (i, means[i] + stds[i] + 250),
                    ha="center", fontsize=12, fontweight="bold", color=EMIS["ink"])
    # oracle reference line
    ax.axhline(d["greedy"]["mean_profit_eur"], color=EMIS["muted"], ls="--", lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([nice[k] for k in order], fontsize=11)
    ax.set_ylabel("mean profit per 24 h episode  (€)")
    ax.set_ylim(0, max(means) + max(stds) + 2200)
    ax.set_title("Battery dispatch — 200 test episodes (2025 Q1)")
    ax.annotate("SAC mean profit exceeds the\nperfect-foresight oracle",
                xy=(3, means[3]), xytext=(1.55, means[3] + 1100),
                fontsize=11.5, color=EMIS["C"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=EMIS["C"], lw=1.4))
    _save(fig, "module_c_profit.png")


# ======================================================================================
# 7. Module C observation-mode ablation (slide 17) — B→C and A→C
# ======================================================================================
def fig_module_c_ablation():
    d = json.load(open(ROOT / "reports/module_c_ablation_results.json"))["results"]
    modes = ["raw", "b", "ba"]
    mode_lbl = ["raw\n(no forecast)", "+ B\n(price q10/50/90)", "+ B + A\n(price + load)"]
    algos = ["PPO", "SAC"]
    acol = {"PPO": EMIS["A"], "SAC": EMIS["C"]}

    def get(mode, algo, field="mean"):
        return d[f"{mode}|{algo}|200000"][field]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    xs = np.arange(len(modes))
    w = 0.36
    for j, algo in enumerate(algos):
        vals = [get(m, algo) for m in modes]
        bars = ax.bar(xs + (j - 0.5) * w, vals, w, color=acol[algo], edgecolor="white",
                      label=algo)
        for i, b in enumerate(bars):
            ax.annotate(f"€{vals[i]:,.0f}", (b.get_x() + b.get_width() / 2, vals[i] + 150),
                        ha="center", fontsize=9.5, fontweight="bold", color=EMIS["ink"])
    ax.set_xticks(xs)
    ax.set_xticklabels(mode_lbl, fontsize=11)
    ax.set_ylabel("mean profit per episode  (€)")
    ax.set_ylim(0, 15500)
    ax.set_title("What the agent sees — observation ablation (SAC / PPO, 200k steps)")
    ax.legend(loc="upper left", fontsize=11)

    sac_raw, sac_b, sac_ba = get("raw", "SAC"), get("b", "SAC"), get("ba", "SAC")
    ax.annotate(f"B forecasts: +{100*(sac_b-sac_raw)/sac_raw:.0f}%",
                xy=(1 + w / 2, sac_b), xytext=(0.0, 13800),
                fontsize=11, color=EMIS["C"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=EMIS["C"], lw=1.3))
    ax.annotate(f"+ A load: +€{sac_ba-sac_b:,.0f}",
                xy=(2 + w / 2, sac_ba), xytext=(1.5, 14600),
                fontsize=11, color=EMIS["imp"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=EMIS["imp"], lw=1.3))
    _save(fig, "module_c_ablation.png")


# ======================================================================================
def main():
    jobs = [
        ("raw price volatility", fig_raw_price),
        ("missingness heatmap", fig_missingness),
        ("SSHB audit status", fig_sshb_audit),
        ("Module B calibration fix", fig_module_b_calib_fix),
        ("Module B leaderboard", fig_module_b_leaderboard),
        ("Module C profit", fig_module_c_profit),
        ("Module C ablation", fig_module_c_ablation),
    ]
    for name, fn in jobs:
        try:
            print(f"[{name}]")
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  !! SKIPPED ({name}): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
