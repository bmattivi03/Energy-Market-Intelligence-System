import numpy as np
import torch

from module_joint.config import JointConfig
from module_joint.data import build_windows
from module_joint.ensemble import SeedEnsemble
from module_joint.model import JointForecasterEstimator


def _est(seed, synthetic_df, fake_load_quantiles):
    cfg = JointConfig(hidden=32, enc_layers=1, dec_layers=1, dropout=0.0, seed=seed)
    wa, scalers = build_windows(synthetic_df, load_quantiles=fake_load_quantiles, fit=True)
    est = JointForecasterEstimator(cfg).setup_from_windows(wa, scalers)
    est.device = torch.device("cpu")
    # perturb weights so members differ
    with torch.no_grad():
        for p in est.model.parameters():
            p.add_(torch.randn_like(p) * 0.01 * seed)
    return est


def test_ensemble_averages_members(synthetic_df, fake_load_quantiles):
    e1 = _est(1, synthetic_df, fake_load_quantiles)
    e2 = _est(2, synthetic_df, fake_load_quantiles)
    p1 = e1.predict_quantiles(synthetic_df, fake_load_quantiles)
    p2 = e2.predict_quantiles(synthetic_df, fake_load_quantiles)
    ens = SeedEnsemble([e1, e2]).predict_quantiles(synthetic_df, fake_load_quantiles)
    expect = (p1["price"] + p2["price"]) / 2
    assert np.allclose(ens["price"].to_numpy(), expect.to_numpy(), atol=1e-5)


def test_ensemble_preserves_non_crossing(synthetic_df, fake_load_quantiles):
    e1 = _est(1, synthetic_df, fake_load_quantiles)
    e2 = _est(2, synthetic_df, fake_load_quantiles)
    ens = SeedEnsemble([e1, e2]).predict_quantiles(synthetic_df, fake_load_quantiles)
    df = ens["price"]
    assert np.all(df["q10"].to_numpy() <= df["q50"].to_numpy() + 1e-5)
    assert np.all(df["q50"].to_numpy() <= df["q90"].to_numpy() + 1e-5)
