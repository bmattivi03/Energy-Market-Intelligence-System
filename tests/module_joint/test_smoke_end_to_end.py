"""End-to-end smoke on a real data slice: train a tiny joint model, predict,
conformalize, and evaluate against a seasonal-naive baseline. Proves the full
pipeline wires together on the actual splits. Marked slow.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from data.loaders import load_split
from module_joint.baselines import SeasonalNaive168
from module_joint.calibrate import conformalize
from module_joint.config import JointConfig
from module_joint.evaluate import compare_price, make_truth
from module_joint.train import train_one

LQ_PATH = "data/module_a/load_quantiles.parquet"


@pytest.mark.slow
def test_end_to_end_on_real_slice():
    lq = pd.read_parquet(LQ_PATH)
    full = load_split("train")
    # a contiguous slice large enough for lookback(168)+horizon(24) windows
    sl = full.iloc[:1600]
    sl = sl[sl.index.isin(lq.index) | True]  # keep order; lq covers these origins
    train = sl.iloc[:1300]
    val = sl.iloc[1300 - 200 :]  # carries 200h lookback context into val

    cfg = JointConfig(
        hidden=32, enc_layers=1, dec_layers=1, dropout=0.0,
        batch_size=64, max_epochs=2, patience=5, seed=0,
    )
    est, history = train_one(cfg, train, val, lq, device=torch.device("cpu"))
    assert len(history["val"]) >= 1 and np.isfinite(history["val"][-1])

    preds = est.predict_quantiles(val, lq, restrict_to=val.index)
    price = preds["price"]
    assert list(price.columns) == ["q10", "q50", "q90"]
    assert np.all(price["q10"].to_numpy() <= price["q50"].to_numpy() + 1e-4)
    assert np.all(price["q50"].to_numpy() <= price["q90"].to_numpy() + 1e-4)

    truth = make_truth(val, price.index, "price")
    cal = conformalize(price, truth, price, alpha=0.20)  # self-cal smoke only
    assert np.allclose(cal["q50"].to_numpy(), price["q50"].to_numpy())

    sn = SeasonalNaive168().fit(train).predict_quantiles(val, restrict_to=val.index)
    res = compare_price(price, truth, baseline_pred=sn["price"])
    assert np.isfinite(res["overall"]["mae"])
    assert np.isfinite(res["overall"]["pinball"])
    assert 0.0 <= res["overall"]["coverage"] <= 1.0
