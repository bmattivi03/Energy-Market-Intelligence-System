"""DL/statistical floors for the backbone bench.

SeasonalNaive168 is the must-beat statistical floor (q50 = value 168h earlier,
bias-corrected, with train-residual quantile bands). dlinear_floor_cfg returns a
JointConfig that turns the model into a plain DLinear floor (linear backbone, no
auxiliary heads, no cross-task pathway), trained through the normal loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import schemas

from .config import HORIZON, LOOKBACK, QUANTILE_COLS, JointConfig


class SeasonalNaive168:
    """Weekly seasonal naive with empirical residual quantile bands."""

    def __init__(self, lag: int = 168, horizon: int = HORIZON):
        self.lag = lag
        self.horizon = horizon
        self.deltas = {}  # target -> (median, lo, hi)

    def fit(self, train_df: pd.DataFrame) -> "SeasonalNaive168":
        for tgt in (schemas.LOAD_COL, schemas.PRICE_COL):
            s = train_df[tgt]
            resid = (s - s.shift(self.lag)).dropna().to_numpy()
            med = float(np.quantile(resid, 0.5))
            lo = float(np.quantile(resid, 0.1)) - med
            hi = float(np.quantile(resid, 0.9)) - med
            self.deltas[tgt] = (med, lo, hi)
        return self

    def predict_quantiles(self, df: pd.DataFrame, *, restrict_to=None) -> dict:
        restrict = set(restrict_to) if restrict_to is not None else None
        out = {}
        for tgt in (schemas.LOAD_COL, schemas.PRICE_COL):
            med, lo, hi = self.deltas[tgt]
            vals = df[tgt].to_numpy(dtype=float)
            rows, idx_pairs = [], []
            last = len(df) - self.horizon - 1
            for p in range(self.lag - 1, last + 1):
                origin_ts = df.index[p]
                if restrict is not None and origin_ts not in restrict:
                    continue
                for h in range(1, self.horizon + 1):
                    q50 = vals[p + h - self.lag] + med
                    rows.append([q50 + lo, q50, q50 + hi])
                    idx_pairs.append((origin_ts, h))
            mi = pd.MultiIndex.from_tuples(idx_pairs, names=["origin_ts", "horizon_h"])
            out[tgt] = pd.DataFrame(rows, index=mi, columns=list(QUANTILE_COLS))
        return out


def dlinear_floor_cfg(**overrides) -> JointConfig:
    base = dict(
        backbone="dlinear",
        aux_residual_load=False,
        aux_renewables=False,
        aux_spike=False,
        load_to_price=False,
        use_load_quantiles=False,
        loss_weighting="fixed",
    )
    base.update(overrides)
    return JointConfig(**base)
