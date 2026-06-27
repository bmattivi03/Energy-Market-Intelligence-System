"""Val-only iteration harness for beating CatBoost on price. Trains a config and
reports price + load MAE per segment on 2024 val vs the CatBoost val bar. The
locked test is NEVER touched here. Iterate by changing flags.

CatBoost val bar (2024, regenerated from the saved Module B model):
  price MAE  h1_6=20.232  h7_18=20.917  h19_24=21.139

Usage:
  PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/joint_val_search.py \
      --tag c1_fundamentals --fundamentals --no-load-to-price --epochs 80 --seeds 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402  before module_b
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
from data.loaders import load_split  # noqa: E402
from module_joint.config import JointConfig, select_device  # noqa: E402
from module_joint.ensemble import train_ensemble  # noqa: E402
from module_joint.train import train_one  # noqa: E402
from module_joint.evaluate import make_truth  # noqa: E402
from module_b import evaluation as ev  # noqa: E402

CB_VAL = {"h1_6": 20.232, "h7_18": 20.917, "h19_24": 21.139}
SEG = {"h1_6": range(1, 7), "h7_18": range(7, 19), "h19_24": range(19, 25)}


def seg_mae(pred, df, col):
    out = {}
    h = pred.index.get_level_values("horizon_h")
    for name, rng in SEG.items():
        p = pred[h.isin(list(rng))]
        y = make_truth(df, p.index, col)
        v = ~y.isna().to_numpy()
        out[name] = ev.mae(y[v].to_numpy(), p[v]["q50"].to_numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--backbone", default="tide")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lookback", type=int, default=168)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--enc-layers", type=int, default=2)
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--fundamentals", action="store_true")
    ap.add_argument("--anchor-residual", action="store_true")
    ap.add_argument("--load-to-price", dest="ltp", action="store_true")
    ap.add_argument("--no-load-to-price", dest="ltp", action="store_false")
    ap.add_argument("--no-aux", action="store_true", help="disable auxiliary heads")
    ap.add_argument("--patience", type=int, default=12)
    ap.set_defaults(ltp=False)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = select_device()
    lq = pd.read_parquet("data/module_a/load_quantiles.parquet")
    train, val = load_split("train"), load_split("val")
    val_ctx = pd.concat([train.tail(args.lookback), val])

    cfg = JointConfig(
        backbone=args.backbone, hidden=args.hidden, lookback=args.lookback,
        lr=args.lr, dropout=args.dropout, enc_layers=args.enc_layers,
        dec_layers=args.dec_layers, max_epochs=args.epochs, patience=args.patience,
        use_load_quantiles=True, load_to_price=args.ltp,
        fundamentals=args.fundamentals, anchor_residual=args.anchor_residual,
        aux_residual_load=not args.no_aux, aux_spike=not args.no_aux,
        aux_renewables=not args.no_aux,
    )
    print(f"[{args.tag}] device={device} seeds={seeds} fundamentals={args.fundamentals} "
          f"anchor={args.anchor_residual} ltp={args.ltp} hidden={args.hidden} "
          f"lookback={args.lookback} lr={args.lr}", flush=True)

    t0 = time.time()
    if len(seeds) == 1:
        cfg.seed = seeds[0]
        est, hist = train_one(cfg, train, val, lq, device=device)
    else:
        est = train_ensemble(cfg, seeds, train, val, lq, device=device)
    dt = time.time() - t0

    preds = est.predict_quantiles(val_ctx, lq, restrict_to=val.index)
    pmae = seg_mae(preds["price"], val_ctx, "price")
    lmae = seg_mae(preds["load"], val_ctx, "load")

    print(f"\n[{args.tag}] trained in {dt:.0f}s", flush=True)
    print(f"[{args.tag}] PRICE MAE vs CatBoost val bar:", flush=True)
    beat = []
    for s in SEG:
        delta = pmae[s] - CB_VAL[s]
        flag = "BEATS" if delta < 0 else "behind"
        beat.append(delta < 0)
        print(f"    {s:7s} DL={pmae[s]:.3f}  CatBoost={CB_VAL[s]:.3f}  "
              f"delta={delta:+.3f}  -> {flag}", flush=True)
    print(f"[{args.tag}] LOAD MAE: " +
          "  ".join(f"{s}={lmae[s]:.1f}" for s in SEG), flush=True)
    print(f"[{args.tag}] VERDICT price h1_6: "
          f"{'BEATS CatBoost' if beat[0] else 'still behind'}", flush=True)


if __name__ == "__main__":
    main()
