import numpy as np

from module_joint.baselines import SeasonalNaive168, dlinear_floor_cfg


def test_seasonal_naive_contract_and_lag(synthetic_df):
    sn = SeasonalNaive168().fit(synthetic_df)
    preds = sn.predict_quantiles(synthetic_df)
    assert set(preds) == {"load", "price"}
    df = preds["price"]
    assert list(df.columns) == ["q10", "q50", "q90"]
    # q50 = lag-168 value + median residual; reconstruct for one row
    origin = df.index.get_level_values("origin_ts")[0]
    p = synthetic_df.index.get_loc(origin)
    med = sn.deltas["price"][0]
    expected = synthetic_df["price"].to_numpy()[p + 1 - 168] + med
    got = df.loc[(origin, 1), "q50"]
    assert np.isclose(got, expected, atol=1e-6)


def test_seasonal_naive_non_crossing(synthetic_df):
    preds = SeasonalNaive168().fit(synthetic_df).predict_quantiles(synthetic_df)
    for df in preds.values():
        assert np.all(df["q10"].to_numpy() <= df["q50"].to_numpy() + 1e-6)
        assert np.all(df["q50"].to_numpy() <= df["q90"].to_numpy() + 1e-6)


def test_dlinear_floor_cfg_is_plain():
    cfg = dlinear_floor_cfg()
    assert cfg.backbone == "dlinear"
    assert not cfg.aux_spike and not cfg.load_to_price and not cfg.use_load_quantiles
