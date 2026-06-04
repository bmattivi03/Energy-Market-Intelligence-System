"""Reward components for the risk-aware battery trading agent.

Three layers:

* :func:`compute_profit` - step P&L from a price and power decision.
* :func:`compute_cvar` - Expected Shortfall (CVaR) at a given tail level.
* :class:`CvarRewardShaper` - stateful shaper that maintains a rolling profit
  history and applies the lambda·CVaR penalty each step.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def compute_profit(price_eur_mwh: float, power_mw: float, dt_h: float = 1.0) -> float:
    """Step profit in euros.

    Positive power = discharge (selling energy) → positive profit.
    Negative power = charge (buying energy) → negative profit (cost).
    """
    return float(price_eur_mwh * power_mw * dt_h)


def compute_cvar(returns: Sequence[float], alpha: float = 0.05) -> float:
    """Expected Shortfall (CVaR) at tail level alpha.

    Returns the mean of the worst-alpha fraction of returns. A negative CVaR
    means the tail is a loss. The reward shaper penalises max(0, -cvar) so
    only loss tails incur a penalty.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    threshold = np.quantile(arr, alpha)
    tail = arr[arr <= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(tail.mean())


@dataclass
class CvarRewardShaper:
    """Stateful reward shaper: step_profit − lambda · max(0, −CVaR).

    Maintains a rolling window of per-step profits. Once the window has at
    least ``min_history`` entries the CVaR penalty is activated; before that
    only raw profit is returned so early training is not dominated by
    sparse-history noise.

    Parameters
    ----------
    lambda_risk:
        Weight on the CVaR penalty (lambda in "Profit − lambda·CVaR").
    alpha:
        Tail level for CVaR - default 0.05 (bottom 5% of profit steps).
    window:
        Rolling-window length in steps. 168 = one week of hourly data,
        matching Module B's longest rolling feature window (price_rmean168h).
    min_history:
        Minimum entries before the CVaR term is activated.
    """

    lambda_risk: float = 0.1
    alpha: float = 0.05
    window: int = 168
    min_history: int = 4
    _history: deque = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.window)

    def reset(self) -> None:
        """Clear rolling history. Call at the start of each episode."""
        self._history.clear()

    def shape(self, step_profit: float) -> float:
        """Return shaped reward for one step."""
        self._history.append(step_profit)
        if len(self._history) < self.min_history:
            return step_profit
        cvar = compute_cvar(list(self._history), alpha=self.alpha)
        penalty = self.lambda_risk * max(0.0, -cvar)
        return step_profit - penalty

    @property
    def current_cvar(self) -> float:
        """CVaR of the current rolling window; nan when history is too short."""
        if len(self._history) < self.min_history:
            return float("nan")
        return compute_cvar(list(self._history), alpha=self.alpha)
