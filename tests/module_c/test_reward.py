"""Tests for module_c.reward.

Covers the pure helpers (`compute_profit`, `compute_cvar`) and the stateful
`CvarRewardShaper`: float return, state reset, `current_cvar` updates, the
lambda·max(0, -CVaR) penalty behaviour on hand-built profit sequences, and
edge cases (empty / short history).
"""

import math

import numpy as np
import pytest

from module_c.reward import CvarRewardShaper, compute_cvar, compute_profit


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_compute_profit_sign_convention():
    # Positive power = discharge (sell) → positive profit.
    assert compute_profit(50.0, 10.0) == pytest.approx(500.0)
    # Negative power = charge (buy) → negative profit (cost).
    assert compute_profit(50.0, -10.0) == pytest.approx(-500.0)
    # dt scaling.
    assert compute_profit(50.0, 10.0, dt_h=0.5) == pytest.approx(250.0)


def test_compute_cvar_empty_returns_zero():
    assert compute_cvar([]) == 0.0


def test_compute_cvar_of_loss_tail_is_negative():
    # Mostly gains with a few large losses → the 5% tail mean is negative.
    returns = [10.0] * 95 + [-100.0] * 5
    cvar = compute_cvar(returns, alpha=0.05)
    assert cvar < 0.0
    assert cvar == pytest.approx(-100.0)


def test_compute_cvar_all_gains_is_positive():
    cvar = compute_cvar([5.0, 6.0, 7.0, 8.0, 9.0], alpha=0.2)
    assert cvar > 0.0


def test_compute_cvar_monotone_in_tail_severity():
    mild = [10.0] * 95 + [-10.0] * 5
    severe = [10.0] * 95 + [-1000.0] * 5
    assert compute_cvar(severe) < compute_cvar(mild)


# ── shaper: basic contract ───────────────────────────────────────────────────

def test_shape_returns_float():
    shaper = CvarRewardShaper()
    out = shaper.shape(12.5)
    assert isinstance(out, float)
    assert out == pytest.approx(12.5)  # raw profit while history < min_history


def test_current_cvar_nan_before_min_history():
    shaper = CvarRewardShaper(min_history=4)
    assert math.isnan(shaper.current_cvar)
    shaper.shape(1.0)
    shaper.shape(2.0)
    # Still below min_history (3 < 4).
    shaper.shape(3.0)
    assert math.isnan(shaper.current_cvar)


def test_current_cvar_becomes_finite_at_min_history():
    shaper = CvarRewardShaper(min_history=4)
    for v in (1.0, 2.0, 3.0, 4.0):
        shaper.shape(v)
    assert np.isfinite(shaper.current_cvar)


def test_shape_returns_raw_profit_until_min_history():
    shaper = CvarRewardShaper(min_history=4, lambda_risk=1.0)
    # First 3 steps: no penalty applied, reward == raw profit.
    assert shaper.shape(-50.0) == pytest.approx(-50.0)
    assert shaper.shape(-50.0) == pytest.approx(-50.0)
    assert shaper.shape(-50.0) == pytest.approx(-50.0)


def test_penalty_reduces_reward_on_loss_tail():
    shaper = CvarRewardShaper(min_history=4, lambda_risk=1.0, alpha=0.25)
    # Build a history dominated by losses so CVaR is clearly negative.
    for v in (-100.0, -100.0, -100.0):
        shaper.shape(v)
    raw_profit = -100.0
    shaped = shaper.shape(raw_profit)
    # With a negative-CVaR tail the penalty is strictly positive → shaped < raw.
    assert shaped < raw_profit
    assert shaper.current_cvar < 0.0


def test_no_penalty_when_tail_is_profitable():
    shaper = CvarRewardShaper(min_history=4, lambda_risk=1.0, alpha=0.25)
    for v in (10.0, 20.0, 30.0):
        shaper.shape(v)
    shaped = shaper.shape(40.0)
    # CVaR positive → max(0, -cvar) == 0 → no penalty, reward == raw profit.
    assert shaper.current_cvar > 0.0
    assert shaped == pytest.approx(40.0)


def test_lambda_scales_penalty():
    profits = (-100.0, -100.0, -100.0, -100.0)

    weak = CvarRewardShaper(min_history=4, lambda_risk=0.1, alpha=0.25)
    strong = CvarRewardShaper(min_history=4, lambda_risk=1.0, alpha=0.25)
    weak_out = strong_out = None
    for v in profits:
        weak_out = weak.shape(v)
        strong_out = strong.shape(v)
    # Larger lambda → larger penalty → smaller (more negative) shaped reward.
    assert strong_out < weak_out


# ── shaper: state management ─────────────────────────────────────────────────

def test_reset_clears_state():
    shaper = CvarRewardShaper(min_history=4)
    for v in (1.0, 2.0, 3.0, 4.0):
        shaper.shape(v)
    assert np.isfinite(shaper.current_cvar)
    shaper.reset()
    assert math.isnan(shaper.current_cvar)
    # After reset the first step is again raw profit (history rebuilt).
    assert shaper.shape(7.0) == pytest.approx(7.0)


def test_window_bounds_history_length():
    shaper = CvarRewardShaper(window=5, min_history=2)
    for v in range(20):
        shaper.shape(float(v))
    assert len(shaper._history) == 5


def test_default_lambda_and_attrs_present():
    shaper = CvarRewardShaper()
    # Attributes referenced by train.py's PPO env factory must exist.
    assert hasattr(shaper, "lambda_risk")
    assert hasattr(shaper, "alpha")
    assert hasattr(shaper, "window")
    assert hasattr(shaper, "min_history")
    assert shaper.lambda_risk == pytest.approx(0.1)
