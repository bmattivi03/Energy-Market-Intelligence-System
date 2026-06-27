import numpy as np
import torch

from module_joint.config import QUANTILE_COLS, JointConfig
from module_joint.data import build_windows
from module_joint.model import JointForecaster, JointForecasterEstimator


def _small_cfg(**kw):
    base = dict(hidden=32, enc_layers=1, dec_layers=1, dropout=0.0)
    base.update(kw)
    return JointConfig(**base)


def test_forward_keys_and_shapes():
    cfg = _small_cfg()
    m = JointForecaster(cfg, n_past_feat=50, n_future_feat=31)
    out = m(torch.randn(4, cfg.lookback, 50), torch.randn(4, cfg.horizon, 31))
    assert out["load"].shape == (4, 24, 3)
    assert out["price"].shape == (4, 24, 3)
    assert out["residual_load"].shape == (4, 24, 1)
    assert out["renewables"].shape == (4, 24, 2)
    assert out["spike"].shape == (4, 24)


def test_forward_quantiles_non_crossing():
    cfg = _small_cfg()
    m = JointForecaster(cfg, 50, 31)
    out = m(torch.randn(8, cfg.lookback, 50), torch.randn(8, cfg.horizon, 31))
    for k in ("load", "price"):
        q = out[k]
        assert torch.all(q[..., 0] <= q[..., 1] + 1e-5)
        assert torch.all(q[..., 1] <= q[..., 2] + 1e-5)


def test_load_to_price_toggle_changes_param_count():
    on = JointForecaster(_small_cfg(load_to_price=True), 50, 31)
    off = JointForecaster(_small_cfg(load_to_price=False), 50, 31)
    n_on = sum(p.numel() for p in on.price_decoder.parameters())
    n_off = sum(p.numel() for p in off.price_decoder.parameters())
    assert n_on > n_off  # price decoder consumes the load head output


def test_predict_quantiles_contract(synthetic_df, fake_load_quantiles):
    cfg = _small_cfg()
    wa, scalers = build_windows(synthetic_df, load_quantiles=fake_load_quantiles, fit=True)
    est = JointForecasterEstimator(cfg).setup_from_windows(wa, scalers)
    est.device = torch.device("cpu")
    preds = est.predict_quantiles(synthetic_df, fake_load_quantiles)
    assert set(preds) == {"load", "price"}
    for tgt, df in preds.items():
        assert list(df.columns) == list(QUANTILE_COLS)
        assert df.index.names == ["origin_ts", "horizon_h"]
        assert set(df.index.get_level_values("horizon_h")) == set(range(1, 25))
        assert np.all(df["q10"].to_numpy() <= df["q50"].to_numpy() + 1e-5)
        assert np.all(df["q50"].to_numpy() <= df["q90"].to_numpy() + 1e-5)
    # price predictions land in a plausible EUR/MWh range (random init, but inverse-scaled)
    assert preds["price"]["q50"].abs().mean() < 1e4


def test_save_load_roundtrip(tmp_path, synthetic_df, fake_load_quantiles):
    cfg = _small_cfg()
    wa, scalers = build_windows(synthetic_df, load_quantiles=fake_load_quantiles, fit=True)
    est = JointForecasterEstimator(cfg).setup_from_windows(wa, scalers)
    est.device = torch.device("cpu")
    p1 = est.predict_quantiles(synthetic_df, fake_load_quantiles)["price"]
    est.save(tmp_path / "m")
    est2 = JointForecasterEstimator.load(tmp_path / "m")
    p2 = est2.predict_quantiles(synthetic_df, fake_load_quantiles)["price"]
    assert np.allclose(p1.to_numpy(), p2.to_numpy(), atol=1e-4)
