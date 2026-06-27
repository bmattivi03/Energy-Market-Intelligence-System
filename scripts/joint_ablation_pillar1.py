"""Phase-2 Pillar 1 ablation: is the joint load->price representation real, or
just load information? Plus the conformal coverage fix (Pillar 3).

Ladder (all TiDE, full aux heads):
  no_load   : use_load_quantiles=False, load_to_price=False
  ext_loadq : use_load_quantiles=True,  load_to_price=False  (Module A quantiles in)
  internal  : use_load_quantiles=True,  load_to_price=True   (joint pathway)

Decisive test: internal must beat ext_loadq on price (DM). Calibration is split
inside 2024 val (Jan-Jun calibrate, Jul-Dec evaluate); the locked test is NOT
touched.

Usage:
  PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/joint_ablation_pillar1.py --epochs 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # tolerate dual libomp

import torch  # noqa: E402  IMPORTANT: import torch before module_b (catboost libomp)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")

from data.loaders import load_split  # noqa: E402
from module_joint.config import JointConfig, select_device  # noqa: E402
from module_joint.train import train_one  # noqa: E402
from module_joint.calibrate import conformalize  # noqa: E402
from module_joint.evaluate import make_truth  # noqa: E402
from module_b import evaluation as ev  # noqa: E402  (loads catboost; after torch)

LQ_PATH = "data/module_a/load_quantiles.parquet"
CAL_END = pd.Timestamp("2024-06-30 23:00", tz="UTC")  # Jan-Jun 2024 calibrates

LADDER = {
    "no_load": dict(use_load_quantiles=False, load_to_price=False),
    "ext_loadq": dict(use_load_quantiles=True, load_to_price=False),
    "internal": dict(use_load_quantiles=True, load_to_price=True),
}


def _metrics(pred, truth):
    y = truth.to_numpy()
    return dict(
        mae=ev.mae(y, pred["q50"].to_numpy()),
        pinball=ev.multi_pinball_loss(y, pred, (0.1, 0.5, 0.9)),
        coverage=ev.coverage(y, pred["q10"].to_numpy(), pred["q90"].to_numpy()),
        winkler=ev.winkler_score(y, pred["q10"].to_numpy(), pred["q90"].to_numpy(), 0.20),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="reports/module_joint_pillar1_ablation")
    args = ap.parse_args()

    device = select_device()
    print(f"[p1] device={device} epochs={args.epochs}", flush=True)
    lq = pd.read_parquet(LQ_PATH)
    train = load_split("train")
    val = load_split("val")
    val_ctx = pd.concat([train.tail(168), val])

    cal_idx = val.index[val.index <= CAL_END]
    eval_idx = val.index[val.index > CAL_END]

    rows, eval_abs_err = [], {}
    for name, flags in LADDER.items():
        t0 = time.time()
        cfg = JointConfig(backbone="tide", max_epochs=args.epochs, **flags)
        est, hist = train_one(cfg, train, val, lq, device=device)
        # one prediction over all val origins, then slice cal / eval
        use_lq = flags["use_load_quantiles"]
        lq_arg = lq if use_lq else None
        preds = est.predict_quantiles(val_ctx, lq_arg, restrict_to=val.index)
        for tgt in ("price", "load"):
            p = preds[tgt]
            h = p.index.get_level_values("origin_ts")
            p_cal = p[h.isin(cal_idx)]
            p_eval = p[h.isin(eval_idx)]
            y_cal = make_truth(val_ctx, p_cal.index, tgt)
            y_eval = make_truth(val_ctx, p_eval.index, tgt)
            raw = _metrics(p_eval, y_eval)
            p_eval_cal = conformalize(p_cal, y_cal, p_eval, alpha=0.20)
            cal = _metrics(p_eval_cal, y_eval)
            rows.append(dict(
                variant=name, target=tgt,
                mae=raw["mae"], pinball=raw["pinball"],
                cov_raw=raw["coverage"], cov_cqr=cal["coverage"],
                winkler_raw=raw["winkler"], winkler_cqr=cal["winkler"],
            ))
            if tgt == "price":
                eval_abs_err[name] = np.abs(
                    y_eval.to_numpy() - p_eval["q50"].to_numpy()
                )
        print(f"[p1] {name}: {time.time()-t0:.1f}s epochs={len(hist['val'])} "
              f"best_val={min(hist['val']):.4f}", flush=True)

    # decisive test: internal vs ext_loadq on price abs error
    decisive = ""
    if "internal" in eval_abs_err and "ext_loadq" in eval_abs_err:
        dm = ev.diebold_mariano(eval_abs_err["internal"], eval_abs_err["ext_loadq"])
        verdict = ("internal SIGNIFICANTLY better" if dm.statistic < 0 and dm.p_value < 0.05
                   else "no significant gain over ext_loadq")
        decisive = (f"DM(internal vs ext_loadq) price: stat={dm.statistic:.3f} "
                    f"p={dm.p_value:.4f} -> {verdict}")
        print("[p1] " + decisive, flush=True)

    lb = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lb.to_parquet(out.with_suffix(".parquet"))
    with open(out.with_suffix(".md"), "w") as f:
        f.write("# module_joint Pillar 1 ablation + conformal (2024 val: Jan-Jun cal, Jul-Dec eval)\n\n")
        f.write(f"device: {device}; epochs: {args.epochs}. Locked test NOT touched.\n\n")
        f.write("Ladder: no_load -> ext_loadq (Module A quantiles) -> internal (joint pathway).\n")
        f.write("The joint pathway is justified only if `internal` beats `ext_loadq`.\n\n")
        f.write(lb.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Decisive test\n\n" + decisive + "\n")
    print(f"[p1] wrote {out.with_suffix('.md')}", flush=True)
    print("\n[p1] price rows:\n" + lb[lb.target == "price"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
