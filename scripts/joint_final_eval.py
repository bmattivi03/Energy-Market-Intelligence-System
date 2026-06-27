"""Phase-3 final evaluation: 4-seed joint ensemble vs the baselines on the LOCKED
2025-Q1 test set. THE one-shot. Trains the chosen config (TiDE, load quantiles in,
internal pathway off), calibrates conformal on 2024 val, evaluates once on test:
  price : vs CatBoost (regenerated q50 from the saved Module B model) and sn168
  load  : vs Module A (its exported load quantiles)
with Diebold-Mariano tests + DM power. Locked test touched ONLY here.

Usage:
  PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/joint_final_eval.py --epochs 80 --seeds 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402  import before module_b (catboost libomp)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")

from data.loaders import load_split  # noqa: E402
from module_joint.config import QUANTILE_COLS, JointConfig, select_device  # noqa: E402
from module_joint.ensemble import train_ensemble  # noqa: E402
from module_joint.calibrate import conformalize  # noqa: E402
from module_joint.evaluate import make_truth  # noqa: E402
import module_b.features as F  # noqa: E402
from module_b.models import CatBoostQuantileForecaster  # noqa: E402
from module_b import evaluation as ev  # noqa: E402

LQ_PATH = "data/module_a/load_quantiles.parquet"
SEG = {"h1_6": range(1, 7), "h7_18": range(7, 19), "h19_24": range(19, 25)}
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")


def module_a_load_contract(lq: pd.DataFrame, origins) -> pd.DataFrame:
    sub = lq[lq.index.isin(origins)]
    H = 24
    q10 = sub[[f"load_q10_h{h}" for h in range(1, H + 1)]].to_numpy()
    q50 = sub[[f"load_q50_h{h}" for h in range(1, H + 1)]].to_numpy()
    q90 = sub[[f"load_q90_h{h}" for h in range(1, H + 1)]].to_numpy()
    n = len(sub)
    idx = pd.MultiIndex.from_arrays(
        [np.repeat(sub.index.to_numpy(), H), np.tile(np.arange(1, H + 1), n)],
        names=["origin_ts", "horizon_h"],
    )
    arr = np.stack([q10.reshape(-1), q50.reshape(-1), q90.reshape(-1)], axis=1)
    return pd.DataFrame(arr, index=idx, columns=list(QUANTILE_COLS))


def catboost_price_contract() -> pd.DataFrame:
    tmeta = json.load(open("checkpoints/module_b/train_meta.json"))
    val, test = load_split("val"), load_split("test")
    ctx = pd.concat([val.tail(400), test])
    feat = F.build_features(ctx, tmeta["bundles"])
    X, _ = F.prepare_supervised(
        feat, past_cols=tmeta["past_cols"], future_cols=tmeta["future_cols"],
        target_col="price",
    )
    X = X[X[F.ORIGIN_COL] >= TEST_START]
    model = CatBoostQuantileForecaster.load("checkpoints/module_b/catboost")
    preds = model.predict_quantiles(X)
    idx = pd.MultiIndex.from_arrays(
        [X[F.ORIGIN_COL].to_numpy(), X[F.HORIZON_COL].to_numpy()],
        names=["origin_ts", "horizon_h"],
    )
    return pd.DataFrame(preds.to_numpy(), index=idx, columns=list(QUANTILE_COLS))


def seg_metrics(pred, df, col):
    out = {}
    h = pred.index.get_level_values("horizon_h")
    for name, rng in SEG.items():
        m = h.isin(list(rng))
        p = pred[m]
        y = make_truth(df, p.index, col)
        valid = ~y.isna().to_numpy()
        p = p[valid]
        y = y[valid].to_numpy()
        out[name] = dict(
            mae=ev.mae(y, p["q50"].to_numpy()),
            pinball=ev.multi_pinball_loss(y, p, (0.1, 0.5, 0.9)),
            coverage=ev.coverage(y, p["q10"].to_numpy(), p["q90"].to_numpy()),
            winkler=ev.winkler_score(y, p["q10"].to_numpy(), p["q90"].to_numpy(), 0.20),
            n=int(valid.sum()),
        )
    return out


def dm_seg(pred_a, pred_b, df, col, seg_name):
    """DM of pred_a vs pred_b on |y-q50| within a segment, on common index."""
    common = pred_a.index.intersection(pred_b.index)
    h = common.get_level_values("horizon_h")
    idx = common[h.isin(list(SEG[seg_name]))]
    y = make_truth(df, idx, col)
    valid = ~y.isna().to_numpy()
    idx = idx[valid]
    y = y[valid].to_numpy()
    ea = np.abs(y - pred_a.loc[idx, "q50"].to_numpy())
    eb = np.abs(y - pred_b.loc[idx, "q50"].to_numpy())
    dm = ev.diebold_mariano(ea, eb)
    sd = float(np.std(ea - eb))
    n = len(idx)
    mde = (1.959964 + 0.8416212) * sd / np.sqrt(max(n, 1))
    return dict(stat=dm.statistic, p=dm.p_value, n=n, mde_mae=mde,
                mean_a=float(ea.mean()), mean_b=float(eb.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--out", default="reports/module_joint_final_eval_v2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = select_device()
    print(f"[final] device={device} epochs={args.epochs} seeds={seeds}", flush=True)
    lq = pd.read_parquet(LQ_PATH)
    train, val, test = load_split("train"), load_split("val"), load_split("test")

    # Winning val-selected config (Cycle 6): feature parity + no aux heads + no
    # internal pathway. Beat CatBoost on val across all segments.
    cfg = JointConfig(backbone="tide", use_load_quantiles=True, load_to_price=False,
                      fundamentals=True, aux_residual_load=False, aux_spike=False,
                      aux_renewables=False, max_epochs=args.epochs, patience=18)

    t0 = time.time()
    ens = train_ensemble(cfg, seeds, train, val, lq, device=device)
    print(f"[final] trained {len(seeds)}-seed ensemble in {time.time()-t0:.1f}s", flush=True)

    val_ctx = pd.concat([train.tail(168), val])
    test_ctx = pd.concat([val.tail(168), test])
    val_pred = ens.predict_quantiles(val_ctx, lq, restrict_to=val.index)
    test_pred = ens.predict_quantiles(test_ctx, lq, restrict_to=test.index)

    # conformal calibrate on val, apply to test (q50 unchanged)
    cal = {}
    for tgt in ("price", "load"):
        vy = make_truth(val_ctx, val_pred[tgt].index, tgt)
        cal[tgt] = conformalize(val_pred[tgt], vy, test_pred[tgt], alpha=0.20)

    # baselines on test
    from module_joint.baselines import SeasonalNaive168
    sn = SeasonalNaive168().fit(train).predict_quantiles(test_ctx, restrict_to=test.index)
    a_load = module_a_load_contract(lq, test.index)
    cb_price = catboost_price_contract()

    rows = []
    def add(model, tgt, pred, col):
        for seg, mt in seg_metrics(pred, test_ctx, col).items():
            rows.append(dict(model=model, target=tgt, segment=seg, **mt))

    add("joint_ensemble", "price", cal["price"], "price")
    add("catboost+cqr", "price", cb_price, "price")
    add("sn168", "price", sn["price"], "price")
    add("joint_ensemble", "load", cal["load"], "load")
    add("module_a", "load", a_load, "load")
    add("sn168", "load", sn["load"], "load")

    dm = {
        "price_vs_catboost": {s: dm_seg(cal["price"], cb_price, test_ctx, "price", s) for s in SEG},
        "price_vs_sn168": {s: dm_seg(cal["price"], sn["price"], test_ctx, "price", s) for s in SEG},
        "load_vs_module_a": {s: dm_seg(cal["load"], a_load, test_ctx, "load", s) for s in SEG},
    }

    lb = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lb.to_parquet(out.with_suffix(".parquet"))
    with open(out.with_suffix(".md"), "w") as f:
        f.write("# module_joint FINAL locked-test eval (2025 Q1)\n\n")
        f.write(f"device: {device}; epochs: {args.epochs}; seeds: {seeds}. "
                f"Winning val-selected config (Cycle 6): TiDE + feature parity "
                f"(Module B fundamentals) + NO aux heads + no internal pathway, "
                f"{len(seeds)}-seed ensemble. Conformal calibrated on 2024 val. "
                f"CatBoost q50 regenerated from the saved Module B model "
                f"(published h1_6 MAE 23.285). This is the second and final "
                f"pre-registered locked-test evaluation.\n\n")
        f.write("## Leaderboard\n\n")
        f.write(lb.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Diebold-Mariano (negative stat = joint better)\n\n")
        for name, segs in dm.items():
            f.write(f"\n### {name}\n\n")
            d = pd.DataFrame(segs).T
            f.write(d.to_markdown(floatfmt=".4f"))
            f.write("\n")
    print(f"[final] wrote {out.with_suffix('.md')}", flush=True)
    pr = lb[(lb.target == "price")].pivot_table(index="segment", columns="model", values="mae")
    print("\n[final] PRICE MAE by segment:\n" + pr.to_string(), flush=True)
    print("\n[final] price DM vs catboost:\n" +
          pd.DataFrame(dm["price_vs_catboost"]).T.to_string(), flush=True)
    ld = lb[(lb.target == "load")].pivot_table(index="segment", columns="model", values="mae")
    print("\n[final] LOAD MAE by segment:\n" + ld.to_string(), flush=True)


if __name__ == "__main__":
    main()
