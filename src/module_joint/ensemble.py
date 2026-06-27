"""Seed ensembling. Averages member predictive quantiles per (origin, horizon,
quantile). Averaging preserves monotonicity, so non-crossing is retained.
Conformal calibration is re-fit on the ensemble output, never inherited.
"""
from __future__ import annotations

import pandas as pd

from .config import JointConfig
from .model import JointForecasterEstimator


class SeedEnsemble:
    def __init__(self, members):
        if not members:
            raise ValueError("SeedEnsemble needs at least one member")
        self.members = list(members)

    def predict_quantiles(self, df, load_quantiles, *, restrict_to=None) -> dict:
        per_member = [
            m.predict_quantiles(df, load_quantiles, restrict_to=restrict_to)
            for m in self.members
        ]
        targets = per_member[0].keys()
        out = {}
        for tgt in targets:
            stacked = pd.concat([pm[tgt] for pm in per_member])
            out[tgt] = stacked.groupby(level=["origin_ts", "horizon_h"]).mean()
        return out


def train_ensemble(cfg: JointConfig, seeds, train_df, val_df, load_quantiles, device=None):
    from .train import train_one

    members = []
    for s in seeds:
        member_cfg = JointConfig(**{**cfg.__dict__, "seed": s})
        est, _ = train_one(member_cfg, train_df, val_df, load_quantiles, device=device)
        members.append(est)
    return SeedEnsemble(members)
