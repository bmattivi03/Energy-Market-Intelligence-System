"""Unit tests for src/data/splits.py CV iterators."""

from __future__ import annotations

import pandas as pd
import pytest

from data.splits import HORIZON, expanding_window_folds, rolling_origin_folds


@pytest.fixture
def long_index() -> pd.DatetimeIndex:
    return pd.date_range("2019-01-01", "2024-12-31 23:00", freq="h", tz="UTC")


def test_expanding_window_folds_yields_5_by_default(long_index: pd.DatetimeIndex) -> None:
    folds = list(expanding_window_folds(long_index, n_folds=5, val_size=pd.Timedelta(days=60)))
    assert len(folds) == 5


def test_expanding_window_folds_train_grows_each_fold(long_index: pd.DatetimeIndex) -> None:
    folds = list(expanding_window_folds(long_index, n_folds=5, val_size=pd.Timedelta(days=60)))
    train_lengths = [(f.train_end - f.train_start) for f in folds]
    for a, b in zip(train_lengths, train_lengths[1:]):
        assert b > a, "train window must grow each fold"


def test_expanding_window_no_train_val_overlap(long_index: pd.DatetimeIndex) -> None:
    folds = list(expanding_window_folds(long_index, n_folds=3, val_size=pd.Timedelta(days=60)))
    for f in folds:
        assert f.val_start > f.train_end, "must have a strictly positive gap"
        gap = f.val_start - f.train_end
        assert gap >= pd.Timedelta(hours=HORIZON), "gap must cover at least the forecast horizon"


def test_rolling_origin_folds_train_size_constant(long_index: pd.DatetimeIndex) -> None:
    folds = list(rolling_origin_folds(
        long_index,
        n_folds=4,
        train_size=pd.Timedelta(days=365),
        val_size=pd.Timedelta(days=60),
    ))
    assert len(folds) == 4
    sizes = [(f.train_end - f.train_start) for f in folds]
    assert all(s == sizes[0] for s in sizes), "rolling-origin train size must be constant"


def test_rolling_origin_train_origin_advances(long_index: pd.DatetimeIndex) -> None:
    folds = list(rolling_origin_folds(
        long_index,
        n_folds=3,
        train_size=pd.Timedelta(days=365),
        val_size=pd.Timedelta(days=60),
    ))
    starts = [f.train_start for f in folds]
    for a, b in zip(starts, starts[1:]):
        assert b > a, "rolling-origin train start must advance each fold"


def test_fold_masks_select_correct_rows(long_index: pd.DatetimeIndex) -> None:
    folds = list(expanding_window_folds(long_index, n_folds=2, val_size=pd.Timedelta(days=30)))
    f = folds[0]
    train_mask = f.train_mask(long_index)
    val_mask = f.val_mask(long_index)
    assert train_mask.sum() > 0
    assert val_mask.sum() > 0
    # Disjoint
    assert (train_mask & val_mask).sum() == 0
    # Order
    assert long_index[train_mask].max() <= f.train_end
    assert long_index[val_mask].min() >= f.val_start


def test_raises_when_insufficient_data() -> None:
    short = pd.date_range("2024-01-01", "2024-01-10", freq="h", tz="UTC")
    with pytest.raises(ValueError, match="Not enough data"):
        list(expanding_window_folds(short, n_folds=5, val_size=pd.Timedelta(days=60)))
