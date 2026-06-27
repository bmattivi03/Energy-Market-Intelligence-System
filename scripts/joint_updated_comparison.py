"""Updated-data fair comparison: retrain BOTH the joint DL model and CatBoost on
2019 -> 2024-09, calibrate conformal on 2024-10..12, evaluate on locked 2025-Q1.
Identical protocol for both models. Module B's files are NOT modified (its classes
are imported and a fresh CatBoost is trained here). The DL winning config from the
val search is used as-is.

Usage:
  PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/joint_updated_comparison.py --seeds 0,1,2,3,4,5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402  before module_b
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
LOOKBACK = 168
TRAIN_END = pd.Timestamp("2024-09-30 23:00", tz="UTC")
CAL_START = pd.Timestamp("2024-10-01 00:00", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01 00:00", tz="UTC")


def seg_metrics(pred, df, col):
    out = {}
    h = pred.index.get_level_values("horizon_h")
    for name, rng in SEG.items():
        p = pred[h.isin(list(rng))]
        y = make_truth(df, p.index, col)
        v = ~y.isna().to_numpy()
        p = p[v]; y = y[v].to_numpy()
        out[name] = dict(
            mae=ev.mae(y, p["q50"].to_numpy()),
            pinball=ev.multi_pinball_loss(y, p, (0.1, 0.5, 0.9)),
            coverage=ev.coverage(y, p["q10"].to_numpy(), p["q90"].to_numpy()),
            winkler=ev.winkler_score(y, p["q10"].to_numpy(), p["q90"].to_numpy(), 0.20),
            n=int(v.sum()),
        )
    return out


def dm_seg(a, b, df, col, seg):
    common = a.index.intersection(b.index)
    h = common.get_level_values("horizon_h")
    idx = common[h.isin(list(SEG[seg]))]
    y = make_truth(df, idx, col)
    v = ~y.isna().to_numpy()
    idx = idx[v]; y = y[v].to_numpy()
    ea = np.abs(y - a.loc[idx, "q50"].to_numpy())
    eb = np.abs(y - b.loc[idx, "q50"].to_numpy())
    dm = ev.diebold_mariano(ea, eb)
    return dict(stat=dm.statistic, p=dm.p_value, n=len(idx),
                mean_a=float(ea.mean()), mean_b=float(eb.mean()))


def catboost_updated(full, meta):
    """Train CatBoost on 2019->2024-09, CQR on 2024-10..12, predict 2025. Returns
    a calibrated test contract DataFrame (origin,horizon)->q10/q50/q90."""
    feat = F.build_features(full, meta["bundles"])
    X, y = F.prepare_supervised(feat, past_cols=meta["past_cols"],
                                future_cols=meta["future_cols"], target_col="price")
    o = X[F.ORIGIN_COL]
    tr, ca, te = o <= TRAIN_END, (o >= CAL_START) & (o < TEST_START), o >= TEST_START
    p = meta["params"]
    model = CatBoostQuantileForecaster(
        mode="per_quantile",
        depth_by_quantile={0.1: 8, 0.5: 6, 0.9: 8},
        l2_by_quantile={0.1: 1.0, 0.5: 3.0, 0.9: 1.0},
        iterations=p.get("iterations", 500), learning_rate=p.get("learning_rate", 0.05),
        early_stopping_rounds=p.get("early_stopping_rounds", 30), random_state=42,
    )
    model.fit(X[tr], y[tr], X_val=X[ca], y_val=y[ca])

    def contract(mask):
        q = model.predict_quantiles(X[mask])
        idx = pd.MultiIndex.from_arrays(
            [X[mask][F.ORIGIN_COL].to_numpy(), X[mask][F.HORIZON_COL].to_numpy()],
            names=["origin_ts", "horizon_h"])
        return pd.DataFrame(q.to_numpy(), index=idx, columns=list(QUANTILE_COLS))

    cal_pred, test_pred = contract(ca), contract(te)
    cal_y = pd.Series(y[ca].to_numpy(), index=cal_pred.index)
    return conformalize(cal_pred, cal_y, test_pred, alpha=0.20)


def module_a_load_contract(lq, origins):
    sub = lq[lq.index.isin(origins)]; H = 24
    parts = {q: sub[[f"load_q{q}_h{h}" for h in range(1, H + 1)]].to_numpy() for q in (10, 50, 90)}
    n = len(sub)
    idx = pd.MultiIndex.from_arrays(
        [np.repeat(sub.index.to_numpy(), H), np.tile(np.arange(1, H + 1), n)],
        names=["origin_ts", "horizon_h"])
    arr = np.stack([parts[10].reshape(-1), parts[50].reshape(-1), parts[90].reshape(-1)], axis=1)
    return pd.DataFrame(arr, index=idx, columns=list(QUANTILE_COLS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--out", default="reports/module_joint_updated_comparison")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = select_device()
    print(f"[upd] device={device} seeds={seeds}", flush=True)

    lq = pd.read_parquet(LQ_PATH)
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    all_2024 = pd.concat([train, val])  # 2019-2024
    train_upd = all_2024[all_2024.index <= TRAIN_END]
    cal_upd = all_2024[all_2024.index >= CAL_START]

    # --- DL: winning config, trained on 2019->2024-09, early-stop/conformal on 2024-Q4 ---
    cfg = JointConfig(backbone="tide", use_load_quantiles=True, load_to_price=False,
                      fundamentals=True, aux_residual_load=False, aux_spike=False,
                      aux_renewables=False, max_epochs=args.epochs, patience=18)
    t0 = time.time()
    ens = train_ensemble(cfg, seeds, train_upd, cal_upd, lq, device=device)
    print(f"[upd] DL {len(seeds)}-seed trained in {time.time()-t0:.0f}s", flush=True)

    cal_ctx = pd.concat([train_upd.tail(LOOKBACK), cal_upd])
    test_ctx = pd.concat([cal_upd.tail(LOOKBACK), test])
    dl_cal = ens.predict_quantiles(cal_ctx, lq, restrict_to=cal_upd.index)
    dl_test = ens.predict_quantiles(test_ctx, lq, restrict_to=test.index)
    dl_price = conformalize(dl_cal["price"], make_truth(cal_ctx, dl_cal["price"].index, "price"),
                            dl_test["price"], alpha=0.20)
    dl_load = conformalize(dl_cal["load"], make_truth(cal_ctx, dl_cal["load"].index, "load"),
                           dl_test["load"], alpha=0.20)

    # --- CatBoost retrained on the same updated data ---
    meta = json.load(open("checkpoints/module_b/catboost/meta.json"))
    meta["bundles"] = json.load(open("checkpoints/module_b/train_meta.json"))["bundles"]
    meta["past_cols"] = json.load(open("checkpoints/module_b/train_meta.json"))["past_cols"]
    meta["future_cols"] = json.load(open("checkpoints/module_b/train_meta.json"))["future_cols"]
    full = pd.concat([train, val, test])
    t0 = time.time()
    cb_price = catboost_updated(full, meta)
    print(f"[upd] CatBoost retrained in {time.time()-t0:.0f}s", flush=True)

    a_load = module_a_load_contract(lq, test.index)  # Module A is fixed (2019-2023 ref)

    rows = []
    def add(model, tgt, pred):
        for s, m in seg_metrics(pred, test_ctx, tgt).items():
            rows.append(dict(model=model, target=tgt, segment=s, **m))
    add("joint_updated", "price", dl_price)
    add("catboost_updated", "price", cb_price)
    add("joint_updated", "load", dl_load)
    add("module_a_2019_23", "load", a_load)

    dm = {s: dm_seg(dl_price, cb_price, test_ctx, "price", s) for s in SEG}

    lb = pd.DataFrame(rows)
    out = Path(args.out)
    lb.to_parquet(out.with_suffix(".parquet"))
    with open(out.with_suffix(".md"), "w") as f:
        f.write("# Updated-data fair comparison (2025-Q1 locked test)\n\n")
        f.write("Both joint-DL and CatBoost trained on 2019->2024-09, conformal on "
                "2024-10..12, tested on 2025-Q1. DL config = feature parity + no aux "
                "+ 6-seed ensemble. Module B files untouched (fresh CatBoost here).\n\n")
        f.write(lb.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Price DM: joint vs CatBoost (negative = joint better)\n\n")
        f.write(pd.DataFrame(dm).T.to_markdown(floatfmt=".4f"))
        f.write("\n")
    print(f"[upd] wrote {out.with_suffix('.md')}", flush=True)
    pr = lb[lb.target == "price"].pivot_table(index="segment", columns="model", values="mae")
    print("\n[upd] PRICE MAE:\n" + pr.to_string(), flush=True)
    print("\n[upd] price DM joint vs catboost:\n" + pd.DataFrame(dm).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
