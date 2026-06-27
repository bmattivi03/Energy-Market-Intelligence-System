import torch

from module_joint.config import JointConfig
from module_joint.train import EarlyStopping, train_one


def _cfg():
    return JointConfig(
        hidden=32, enc_layers=1, dec_layers=1, dropout=0.0,
        batch_size=32, max_epochs=2, patience=5, seed=0,
    )


def test_early_stopping_restores_best():
    m = torch.nn.Linear(2, 2)
    es = EarlyStopping(patience=2)
    assert es.step(1.0, m) is False
    assert es.step(0.5, m) is False  # improved
    assert es.step(0.6, m) is False  # 1 worse
    assert es.step(0.6, m) is True  # 2 worse -> stop
    assert es.best == 0.5 and es.best_state is not None


def test_train_one_runs_and_predicts(synthetic_df, fake_load_quantiles):
    cfg = _cfg()
    train = synthetic_df.iloc[:300]
    val = synthetic_df.iloc[132:]  # carries lookback context for val origins
    est, history = train_one(
        cfg, train, val, fake_load_quantiles, device=torch.device("cpu")
    )
    assert len(history["val"]) >= 1
    assert all(v == v for v in history["val"])  # no NaN
    preds = est.predict_quantiles(val, fake_load_quantiles, restrict_to=val.index)
    assert set(preds) == {"load", "price"}
    assert list(preds["price"].columns) == ["q10", "q50", "q90"]


def test_train_one_deterministic(synthetic_df, fake_load_quantiles):
    cfg = _cfg()
    train = synthetic_df.iloc[:300]
    val = synthetic_df.iloc[132:]
    e1, h1 = train_one(cfg, train, val, fake_load_quantiles, device=torch.device("cpu"))
    e2, h2 = train_one(cfg, train, val, fake_load_quantiles, device=torch.device("cpu"))
    assert h1["val"] == h2["val"]
