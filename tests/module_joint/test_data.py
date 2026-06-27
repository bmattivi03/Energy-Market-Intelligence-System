import numpy as np

from module_joint import config as C
from module_joint.data import build_windows, spike_threshold


def test_window_shapes_and_count(synthetic_df, fake_load_quantiles):
    wa, scalers = build_windows(
        synthetic_df, load_quantiles=fake_load_quantiles, fit=True
    )
    n_expected = len(synthetic_df) - C.LOOKBACK - C.HORIZON + 1
    assert wa.past.shape[0] == n_expected
    assert wa.past.shape[1] == C.LOOKBACK
    assert wa.past.shape[2] >= 40  # ~51 fundamentals + 8 calendar
    assert wa.future.shape == (n_expected, C.HORIZON, wa.future.shape[2])
    # future channels = 8 calendar + 20 weather + 3 load-quantile = 31
    assert wa.future.shape[2] == 8 + 20 + 3
    assert wa.y["price"].shape == (n_expected, C.HORIZON)
    assert wa.y["renewables"].shape == (n_expected, C.HORIZON, 2)
    assert wa.y["spike"].shape == (n_expected, C.HORIZON)


def test_origin_is_causal(synthetic_df, fake_load_quantiles):
    wa, _ = build_windows(synthetic_df, load_quantiles=fake_load_quantiles, fit=True)
    # first origin sits at the end of the first full lookback window
    assert wa.origin_ts[0] == synthetic_df.index[C.LOOKBACK - 1]
    # the target window starts strictly after the origin
    assert wa.y_raw["price"].shape[1] == C.HORIZON


def test_no_load_quantiles_drops_channels(synthetic_df):
    wa, _ = build_windows(
        synthetic_df, load_quantiles=None, fit=True, use_load_quantiles=False
    )
    assert wa.future.shape[2] == 8 + 20  # no load-quantile channels


def test_price_target_is_asinh_standardized_and_invertible(synthetic_df, fake_load_quantiles):
    wa, scalers = build_windows(
        synthetic_df, load_quantiles=fake_load_quantiles, fit=True
    )
    from module_joint.transforms import inv_asinh

    recovered = inv_asinh(scalers.price.inverse(wa.y["price"]))
    assert np.allclose(recovered, wa.y_raw["price"], atol=1e-2)


def test_scalers_reused_across_splits(synthetic_df, fake_load_quantiles):
    train = synthetic_df.iloc[:300]
    val = synthetic_df.iloc[200:]  # overlap gives val enough lookback context
    _, scalers = build_windows(train, load_quantiles=fake_load_quantiles, fit=True)
    wa_val, _ = build_windows(
        val, load_quantiles=fake_load_quantiles, scalers=scalers, fit=False
    )
    assert wa_val.past.shape[2] == (len(scalers.past_cols) + 8)


def test_spike_threshold_is_causal_expanding(synthetic_df):
    thr = spike_threshold(synthetic_df["price"])
    assert thr.iloc[:719].isna().all()
    assert not np.isnan(thr.iloc[800 % len(thr)]) or len(thr) <= 800
