"""Phase-1 single-task backbone bench for module_joint (run on real data).

Trains the joint model with each candidate backbone on the full train split and
evaluates load + price on the 2024 validation split, alongside the SeasonalNaive
floor. Writes a leaderboard to reports/. Does NOT touch the locked test split.

Usage:
  PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 \
      python scripts/joint_backbone_bench.py --epochs 60 --backbones tide,dlinear,nhitsx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

from data.loaders import load_split  # noqa: E402
from module_joint.baselines import SeasonalNaive168  # noqa: E402
from module_joint.config import JointConfig, select_device  # noqa: E402
from module_joint.evaluate import compare_load, compare_price, make_truth  # noqa: E402
from module_joint.train import train_one  # noqa: E402

LQ_PATH = "data/module_a/load_quantiles.parquet"


def _rows(tag, model, price_res, load_res):
    out = []
    for seg in ("overall", "h1_6", "h7_18", "h19_24"):
        pm = price_res["overall"] if seg == "overall" else price_res["segments"].get(seg)
        lm = load_res["overall"] if seg == "overall" else load_res["segments"].get(seg)
        if pm is None:
            continue
        row = {
            "model": model,
            "segment": seg,
            "price_mae": pm["mae"],
            "price_pinball": pm["pinball"],
            "price_cov": pm["coverage"],
            "price_winkler": pm["winkler"],
            "load_mae": lm["mae"] if lm else float("nan"),
            "load_pinball": lm["pinball"] if lm else float("nan"),
            "load_cov": lm["coverage"] if lm else float("nan"),
        }
        if "dm_vs_baseline" in price_res and seg == "overall":
            row["price_dm_stat"] = price_res["dm_vs_baseline"]["statistic"]
            row["price_dm_p"] = price_res["dm_vs_baseline"]["p_value"]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--backbones", default="tide,dlinear,nhitsx")
    ap.add_argument("--out", default="reports/module_joint_backbone_bench")
    args = ap.parse_args()

    device = select_device()
    print(f"[bench] device={device} epochs={args.epochs}", flush=True)

    lq = pd.read_parquet(LQ_PATH)
    train = load_split("train")
    val = load_split("val")
    val_ctx = pd.concat([train.tail(168), val])
    price_truth = None  # filled per prediction index

    rows = []

    # SeasonalNaive(168) floor
    t0 = time.time()
    sn = SeasonalNaive168().fit(train).predict_quantiles(val_ctx, restrict_to=val.index)
    p_tr = make_truth(val_ctx, sn["price"].index, "price")
    l_tr = make_truth(val_ctx, sn["load"].index, "load")
    rows += _rows("sn168", "sn168", compare_price(sn["price"], p_tr), compare_load(sn["load"], l_tr))
    sn_price = sn["price"]
    print(f"[bench] sn168 done in {time.time()-t0:.1f}s", flush=True)

    for bb in [b.strip() for b in args.backbones.split(",") if b.strip()]:
        t0 = time.time()
        cfg = JointConfig(backbone=bb, max_epochs=args.epochs)
        est, hist = train_one(cfg, train, val, lq, device=device)
        preds = est.predict_quantiles(val_ctx, lq, restrict_to=val.index)
        p_tr = make_truth(val_ctx, preds["price"].index, "price")
        l_tr = make_truth(val_ctx, preds["load"].index, "load")
        pres = compare_price(preds["price"], p_tr, baseline_pred=sn_price)
        lres = compare_load(preds["load"], l_tr)
        rows += _rows(bb, bb, pres, lres)
        ckpt = f"checkpoints/module_joint/bench_{bb}"
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        est.save(ckpt)
        print(
            f"[bench] {bb}: {time.time()-t0:.1f}s  epochs_ran={len(hist['val'])}  "
            f"best_val_pinball={min(hist['val']):.4f}  "
            f"price_MAE(h1_6)={pres['segments']['h1_6']['mae']:.2f}  "
            f"saved={ckpt}",
            flush=True,
        )

    lb = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lb.to_parquet(out.with_suffix(".parquet"))
    with open(out.with_suffix(".md"), "w") as f:
        f.write("# module_joint backbone bench (2024 val)\n\n")
        f.write(f"device: {device}; epochs: {args.epochs}; ")
        f.write("baselines reported, not retrained. Locked test NOT touched.\n\n")
        f.write(lb.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(f"[bench] wrote {out.with_suffix('.md')}", flush=True)
    # headline: best price MAE on h1-6
    h16 = lb[lb.segment == "h1_6"].sort_values("price_mae")
    print("\n[bench] price MAE h1-6 ranking:\n" + h16[["model", "price_mae", "price_cov"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
