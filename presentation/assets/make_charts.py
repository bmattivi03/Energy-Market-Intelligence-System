#!/usr/bin/env python3
"""Raster charts for the HTML deck (the two plots that genuinely need pixels).

Everything else in the deck (bar charts, diagrams) is drawn natively in HTML/CSS/SVG.
These two are rendered here, restyled for the warm 'field-report' theme: transparent
background (the slide card shows through), minimal chrome, high DPI.

    python3 presentation/assets/make_charts.py

Reads git-ignored local data (present after the pipeline run); outputs PNGs into
presentation/assets/img/. Re-run only if the underlying data changes.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "img"; OUT.mkdir(parents=True, exist_ok=True)

INK   = "#14181C"
MUTED = "#6B7178"
GRID  = "#E4DED2"
PAPER = "#F6F2EA"
TEAL  = "#0E7C86"
AMBER = "#D9772B"
GREEN = "#2E8B57"
RED   = "#C0392B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "savefig.dpi": 200, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.transparent": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def price_volatility():
    df = pd.read_parquet(ROOT / "data/splits/test.parquet")
    s = df["price"].dropna()
    hi = s.quantile(0.95)
    fig, ax = plt.subplots(figsize=(12, 3.7))
    ax.fill_between(s.index, s.values, 0, where=(s.values > 0),
                    color=AMBER, alpha=0.08, linewidth=0)
    ax.plot(s.index, s.values, color=AMBER, lw=0.9)
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 4)))
    sp = s[s > hi]; ng = s[s < 0]
    ax.scatter(sp.index, sp.values, s=11, color=RED, zorder=3, label=f"spikes  > {hi:.0f}")
    ax.scatter(ng.index, ng.values, s=11, color=TEAL, zorder=3, label="negative prices")
    ax.set_ylabel("€ / MWh", color=MUTED)
    ax.set_xlim(s.index.min(), s.index.max())
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.annotate(f"max  {s.max():.0f}", (s.idxmax(), s.max()), color=RED, fontsize=13,
                xytext=(8, -1), textcoords="offset points", fontweight="bold")
    ax.annotate(f"min  {s.min():.0f}", (s.idxmin(), s.min()), color=TEAL, fontsize=13,
                xytext=(8, 2), textcoords="offset points", fontweight="bold")
    ax.legend(loc="upper right", fontsize=12, handletextpad=0.4, borderpad=0.3)
    fig.autofmt_xdate(rotation=0, ha="center")
    for lbl in ax.get_xticklabels():
        lbl.set_color(MUTED)
    fig.savefig(OUT / "price_volatility.png")
    plt.close(fig)
    print("wrote", (OUT / "price_volatility.png").relative_to(ROOT))


def missingness():
    mask = pd.read_parquet(ROOT / "data/processed/emis_mask.parquet")
    weekly = mask.astype(float).resample("W").mean()
    cols = list(mask.columns)
    M = weekly[cols].to_numpy().T
    nrows, nweeks = M.shape
    idx = weekly.index
    cmap = LinearSegmentedColormap.from_list("miss", ["#EFE9DA", AMBER, RED])

    def wk(d):
        return int(idx.searchsorted(pd.Timestamp(d, tz="UTC")))

    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=[0, nweeks, nrows, 0], interpolation="nearest")
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels([c.replace("_", " ") for c in cols], fontsize=7.5, color=MUTED)
    years = list(range(2019, 2026))
    ax.set_xticks([wk(f"{y}-01-01") for y in years])
    ax.set_xticklabels(years, color=MUTED, fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    def row(n):
        return cols.index(n) + 0.5 if n in cols else None

    rc = row("gen_fossil_coal_gas")
    if rc is not None:
        ax.annotate("coal-gas: no ENTSO-E data 2019–22",
                    xy=(wk("2020-09-01"), rc), xytext=(wk("2021-02-01"), rc - 6.5),
                    fontsize=12, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    rn = row("gen_nuclear")
    if rn is not None:
        ax.annotate("nuclear → 0 after 2023-04-15",
                    xy=(wk("2023-05-01"), rn), xytext=(wk("2019-07-01"), rn + 7),
                    fontsize=12, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    fig.savefig(OUT / "missingness_heatmap.png")
    plt.close(fig)
    print("wrote", (OUT / "missingness_heatmap.png").relative_to(ROOT))


if __name__ == "__main__":
    price_volatility()
    missingness()
