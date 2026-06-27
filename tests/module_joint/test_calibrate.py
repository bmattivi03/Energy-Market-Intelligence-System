import numpy as np
import pandas as pd

from module_joint.calibrate import barber_level, conformalize


def _frame(origins, q10, q50, q90):
    idx = pd.MultiIndex.from_product([origins, [1]], names=["origin_ts", "horizon_h"])
    return pd.DataFrame({"q10": q10, "q50": q50, "q90": q90}, index=idx)


def test_barber_level_formula():
    assert barber_level(99, 0.20) == np.ceil(100 * 0.8) / 99


def test_conformalize_q50_unchanged_and_widens():
    origins = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
    val = _frame(origins, np.full(50, -1.0), np.zeros(50), np.full(50, 1.0))
    # actuals frequently fall outside [-1, 1] so delta > 0
    val_y = pd.Series(np.linspace(-3, 3, 50), index=val.index)
    test_origins = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    test = _frame(test_origins, np.full(10, -1.0), np.zeros(10), np.full(10, 1.0))

    cal = conformalize(val, val_y, test, alpha=0.20)
    assert np.allclose(cal["q50"].to_numpy(), test["q50"].to_numpy())  # median untouched
    assert np.all(cal["q10"].to_numpy() <= test["q10"].to_numpy())  # widened down
    assert np.all(cal["q90"].to_numpy() >= test["q90"].to_numpy())  # widened up
    assert (cal["q90"] - cal["q10"]).iloc[0] > (test["q90"] - test["q10"]).iloc[0]
