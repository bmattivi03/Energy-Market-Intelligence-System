"""Time-series cross-validation iterators.

Two strategies, both aware of horizon-induced leakage:

* :func:`rolling_origin_folds` — each fold has a fixed-size training window
  followed by a contiguous validation window, sliding forward.
* :func:`expanding_window_folds` — each fold expands the training window from
  a common origin; validation window slides forward by ``val_size``.

Both insert a ``gap`` between train and val to prevent the model from learning
that the validation window starts at the next timestamp (lookahead leakage
through e.g. lag-24 features that span fold boundaries). For day-ahead
forecasting, ``gap`` should be at least ``HORIZON``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

HORIZON  = 24   # day-ahead forecast horizon (hours)
LOOKBACK = 168  # one week — default sequence input window for LSTM models


@dataclass(frozen=True)
class Fold:
    """A single train/val fold for time-series CV."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def train_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        return (index >= self.train_start) & (index <= self.train_end)

    def val_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        return (index >= self.val_start) & (index <= self.val_end)


def expanding_window_folds(
    index: pd.DatetimeIndex,
    *,
    n_folds: int = 5,
    val_size: pd.Timedelta = pd.Timedelta(days=90),
    gap: pd.Timedelta = pd.Timedelta(hours=HORIZON),
    initial_train: pd.Timedelta | None = None,
) -> Iterator[Fold]:
    """Yield expanding-window folds."""
    index = pd.DatetimeIndex(index).sort_values()
    total = index.max() - index.min()
    if initial_train is None:
        initial_train = total - n_folds * val_size - gap
    if initial_train <= pd.Timedelta(0):
        raise ValueError(
            f"Not enough data for {n_folds} folds of size {val_size} with "
            f"gap {gap}. Total span: {total}"
        )
    for k in range(n_folds):
        train_end = index.min() + initial_train + k * val_size
        val_start = train_end + gap
        val_end = val_start + val_size
        if val_end > index.max():
            return
        yield Fold(
            fold_id=k,
            train_start=index.min(),
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        )


def rolling_origin_folds(
    index: pd.DatetimeIndex,
    *,
    n_folds: int = 5,
    train_size: pd.Timedelta = pd.Timedelta(days=365 * 2),
    val_size: pd.Timedelta = pd.Timedelta(days=90),
    gap: pd.Timedelta = pd.Timedelta(hours=HORIZON),
) -> Iterator[Fold]:
    """Yield fixed-window rolling-origin folds."""
    index = pd.DatetimeIndex(index).sort_values()
    total = index.max() - index.min()
    needed = train_size + gap + val_size + (n_folds - 1) * val_size
    if total < needed:
        raise ValueError(
            f"Need {needed} for {n_folds} rolling folds; have {total}."
        )
    for k in range(n_folds):
        train_start = index.min() + k * val_size
        train_end = train_start + train_size
        val_start = train_end + gap
        val_end = val_start + val_size
        if val_end > index.max():
            return
        yield Fold(
            fold_id=k,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        )
