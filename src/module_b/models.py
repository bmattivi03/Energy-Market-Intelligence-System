"""Quantile forecasters for day-ahead electricity prices.

One file holding the Module B model zoo: the :class:`BaseQuantileForecaster`
ABC, three baselines (Naive, SeasonalNaive, LEAR), and four classical
wrappers (CatBoost, LightGBM, Ridge, ElasticNet). All models share the same
fit/predict_quantiles interface and consume the flat (origin, horizon) row
layout produced by :func:`module_b.features.prepare_supervised`.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from module_b.features import HORIZON_COL, ORIGIN_COL, TARGET_COL


# ============================================================ base


class BaseQuantileForecaster(ABC):
    """Abstract base class for quantile forecasters."""

    name: str

    def __init__(self, *, name: str | None = None, quantiles: Sequence[float] = (0.1, 0.5, 0.9)):
        self.name = name or type(self).__name__
        self.quantiles = tuple(quantiles)

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseQuantileForecaster":
        """Fit the model. Returns self for chaining."""

    @abstractmethod
    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict every quantile in ``self.quantiles``."""

    def predict_point(self, X: pd.DataFrame) -> pd.Series:
        q = self.predict_quantiles(X)
        return q[f"q{int(round(0.5 * 100))}"]

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseQuantileForecaster":
        """Load a previously saved model."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, quantiles={self.quantiles!r})"


def quantile_columns(quantiles: Sequence[float]) -> list[str]:
    """Canonical column names for a list of quantile levels."""
    return [f"q{int(round(q * 100))}" for q in quantiles]


# ============================================================ baselines


# ---------------------------------------------------------------- residual-quantile helper

@dataclass
class ResidualQuantileEstimator:
    """Per-horizon empirical quantiles of training residuals."""

    quantiles: tuple[float, ...]
    by_horizon: dict[int, np.ndarray] | None = None
    pooled: np.ndarray | None = None

    def fit(self, residuals: pd.Series, horizons: pd.Series | None = None) -> "ResidualQuantileEstimator":
        if horizons is not None:
            self.by_horizon = {
                int(h): np.quantile(residuals[horizons == h].dropna(), self.quantiles)
                for h in horizons.unique()
            }
        self.pooled = np.quantile(residuals.dropna(), self.quantiles)
        return self

    def predict_quantiles(
        self, point: pd.Series, horizons: pd.Series | None = None
    ) -> pd.DataFrame:
        n = len(point)
        cols = [f"q{int(round(q * 100))}" for q in self.quantiles]
        out = pd.DataFrame(0.0, index=point.index, columns=cols)
        if self.by_horizon is not None and horizons is not None:
            arr = np.zeros((n, len(self.quantiles)))
            h_arr = horizons.to_numpy()
            for h, q_offsets in self.by_horizon.items():
                mask = h_arr == h
                if mask.any():
                    arr[mask] = q_offsets
            out.iloc[:, :] = point.to_numpy()[:, None] + arr
        else:
            assert self.pooled is not None
            out.iloc[:, :] = point.to_numpy()[:, None] + self.pooled[None, :]
        return out


# ---------------------------------------------------------------- Naive

class NaiveForecaster(BaseQuantileForecaster):
    """``ŷ_{t+h} = y_t`` (read from an anchor column, default ``price_lag1h``)."""

    def __init__(
        self,
        *,
        anchor_column: str = "price_lag1h",
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__(name="naive", quantiles=quantiles)
        self.anchor_column = anchor_column
        self._residual_est: ResidualQuantileEstimator | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight=None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "NaiveForecaster":
        if self.anchor_column not in X.columns:
            raise KeyError(
                f"NaiveForecaster needs the '{self.anchor_column}' column in X "
                "(typically price_lag1h provided by the lags bundle)."
            )
        residuals = y - X[self.anchor_column]
        horizons = X[HORIZON_COL] if HORIZON_COL in X.columns else None
        self._residual_est = ResidualQuantileEstimator(quantiles=self.quantiles)
        self._residual_est.fit(residuals, horizons=horizons)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._residual_est is None:
            raise RuntimeError("must call fit() before predict_quantiles()")
        point = X[self.anchor_column]
        horizons = X[HORIZON_COL] if HORIZON_COL in X.columns else None
        out = self._residual_est.predict_quantiles(point, horizons=horizons)
        out.index = X.index
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "anchor_column": self.anchor_column,
                    "quantiles": self.quantiles,
                    "residual_est": self._residual_est,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "NaiveForecaster":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        m = cls(anchor_column=state["anchor_column"], quantiles=state["quantiles"])
        m._residual_est = state["residual_est"]
        return m


# ---------------------------------------------------------------- SeasonalNaive

class SeasonalNaiveForecaster(BaseQuantileForecaster):
    """``ŷ_{t+h} = price[t+h − season]``.

    Requires the global price series via ``price_series=…`` at fit time so we
    can index by ``target_ts``.
    """

    def __init__(
        self,
        *,
        season_hours: int = 168,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__(name=f"seasonal_naive_{season_hours}h", quantiles=quantiles)
        self.season_hours = season_hours
        self._price_series: pd.Series | None = None
        self._residual_est: ResidualQuantileEstimator | None = None

    def _seasonal_pred(self, X: pd.DataFrame) -> pd.Series:
        if self._price_series is None:
            raise RuntimeError("set ``price_series`` via fit() first")
        target_ts = X[TARGET_COL]
        lookup_ts = target_ts - pd.Timedelta(hours=self.season_hours)
        return self._price_series.reindex(lookup_ts).set_axis(X.index)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight=None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        price_series: Optional[pd.Series] = None,
    ) -> "SeasonalNaiveForecaster":
        if price_series is None:
            raise ValueError(
                "SeasonalNaiveForecaster.fit needs the global price series via "
                "price_series=… so we can look up price[target_ts − season]."
            )
        self._price_series = price_series
        point = self._seasonal_pred(X)
        residuals = y - point
        horizons = X[HORIZON_COL] if HORIZON_COL in X.columns else None
        self._residual_est = ResidualQuantileEstimator(quantiles=self.quantiles).fit(
            residuals, horizons=horizons
        )
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._residual_est is None:
            raise RuntimeError("must call fit() before predict_quantiles()")
        point = self._seasonal_pred(X)
        point = point.ffill().bfill()
        horizons = X[HORIZON_COL] if HORIZON_COL in X.columns else None
        out = self._residual_est.predict_quantiles(point, horizons=horizons)
        out.index = X.index
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "season_hours": self.season_hours,
                    "quantiles": self.quantiles,
                    "price_series": self._price_series,
                    "residual_est": self._residual_est,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "SeasonalNaiveForecaster":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        m = cls(season_hours=state["season_hours"], quantiles=state["quantiles"])
        m._price_series = state["price_series"]
        m._residual_est = state["residual_est"]
        return m


# ---------------------------------------------------------------- LEAR

# ============================================================ classical


# ---------------------------------------------------------------- feature helpers

BOOKKEEPING_COLS: tuple[str, ...] = (ORIGIN_COL, TARGET_COL)


def select_feature_columns(X: pd.DataFrame, *, drop: Iterable[str] = ()) -> list[str]:
    drop_set = set(BOOKKEEPING_COLS) | set(drop)
    return [c for c in X.columns if c not in drop_set]


def to_array(X: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return X[feature_names].to_numpy(dtype=np.float32)


def detect_categorical_indices(
    X: pd.DataFrame, feature_names: list[str], *, max_unique: int = 2,
) -> list[int]:
    """Find indices of binary 0/1 feature columns.

    Returned positions are into ``feature_names`` (i.e. matching the array
    produced by :func:`to_array`). LightGBM and CatBoost both accept these
    indices via ``categorical_feature`` / ``cat_features`` so they can build
    proper split rules instead of treating the column as a continuous
    threshold. Default ``max_unique=2`` catches the binary flag features
    produced by :mod:`module_b.features` (is_weekend, is_holiday_DE,
    is_dst_transition, crisis_2022_flag, is_high_residual_load,
    is_renewable_scarcity).
    """
    cat_idx: list[int] = []
    for i, col in enumerate(feature_names):
        if col == HORIZON_COL:
            cat_idx.append(i)
            continue
        s = X[col]
        if s.dtype.kind not in {"f", "i"}:
            continue
        n_unique = s.dropna().nunique()
        if 1 < n_unique <= max_unique:
            cat_idx.append(i)
    return cat_idx


# ---------------------------------------------------------------- CatBoost

class CatBoostQuantileForecaster(BaseQuantileForecaster):
    """CatBoost quantile forecaster.

    Two operating modes:

    * ``mode="per_quantile"`` (default): one CatBoostRegressor per quantile,
      each with its own ``Quantile:alpha=X`` loss. Per-quantile ``depth`` and
      ``l2_leaf_reg`` are tunable so the tails can use deeper trees than the
      median — important for heavy-tailed price data where ``MultiQuantile``
      systematically under-fits the upper tail (see
      ``reports/module_b_catboost_calibration_diagnosis.md``).
    * ``mode="multi"``: single CatBoostRegressor with
      ``MultiQuantile:alpha=q1,q2,…`` loss. Legacy behaviour; kept so saved
      checkpoints from prior sessions (``outputs/B5_catboost_*``) can still
      be loaded.
    """

    _DEFAULT_DEPTH_BY_QUANTILE = {0.1: 8, 0.5: 6, 0.9: 8}
    _DEFAULT_L2_BY_QUANTILE = {0.1: 1.0, 0.5: 3.0, 0.9: 1.0}

    def __init__(
        self,
        *,
        params: Optional[dict] = None,
        iterations: int = 5000,
        early_stopping_rounds: int = 100,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        depth_by_quantile: Optional[dict[float, int]] = None,
        l2_by_quantile: Optional[dict[float, float]] = None,
        learning_rate: float = 0.05,
        random_state: int = 42,
        mode: str = "per_quantile",
        **catboost_kwargs,
    ):
        super().__init__(name="catboost_q", quantiles=quantiles)
        if mode not in {"per_quantile", "multi"}:
            raise ValueError(f"mode must be 'per_quantile' or 'multi'; got {mode!r}")
        self.mode = mode
        self.depth_by_quantile = depth_by_quantile or {
            q: self._DEFAULT_DEPTH_BY_QUANTILE.get(round(q, 2), 6) for q in quantiles
        }
        self.l2_by_quantile = l2_by_quantile or {
            q: self._DEFAULT_L2_BY_QUANTILE.get(round(q, 2), 3.0) for q in quantiles
        }
        common_params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "random_seed": random_state,
            "verbose": False,
            "early_stopping_rounds": early_stopping_rounds,
            "border_count": 254,
            "thread_count": -1,
            **(params or {}),
            **catboost_kwargs,
        }
        if mode == "multi":
            alpha_str = ",".join(f"{q:.2f}" for q in quantiles)
            self.params = {
                "loss_function": f"MultiQuantile:alpha={alpha_str}",
                "eval_metric": f"MultiQuantile:alpha={alpha_str}",
                "depth": 6,
                "l2_leaf_reg": 3.0,
                **common_params,
            }
        else:
            self.params = common_params
        self._model: CatBoostRegressor | None = None
        self._models: dict[float, CatBoostRegressor] = {}
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight=None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "CatBoostQuantileForecaster":
        self._feature_names = select_feature_columns(X)
        X_arr = to_array(X, self._feature_names)
        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = Pool(to_array(X_val, self._feature_names), y_val.to_numpy())
        if self.mode == "multi":
            train_pool = Pool(X_arr, y.to_numpy(), weight=sample_weight)
            self._model = CatBoostRegressor(**self.params)
            self._model.fit(
                train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None,
            )
            return self
        # Per-quantile mode: one booster per α, with α-specific depth + l2.
        for q in self.quantiles:
            params_q = {
                **self.params,
                "loss_function": f"Quantile:alpha={q:.2f}",
                "eval_metric": f"Quantile:alpha={q:.2f}",
                "depth": int(self.depth_by_quantile.get(q, 6)),
                "l2_leaf_reg": float(self.l2_by_quantile.get(q, 3.0)),
            }
            train_pool = Pool(X_arr, y.to_numpy(), weight=sample_weight)
            booster = CatBoostRegressor(**params_q)
            booster.fit(
                train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None,
            )
            self._models[float(q)] = booster
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        X_arr = to_array(X, self._feature_names)
        if self.mode == "multi":
            if self._model is None:
                raise RuntimeError("must call fit() before predict_quantiles()")
            preds = self._model.predict(X_arr)
            cols = [f"q{int(round(q * 100))}" for q in self.quantiles]
            return pd.DataFrame(preds, index=X.index, columns=cols)
        if not self._models:
            raise RuntimeError("must call fit() before predict_quantiles()")
        out = pd.DataFrame(index=X.index)
        for q, booster in self._models.items():
            out[f"q{int(round(q * 100))}"] = booster.predict(X_arr)
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.mode == "multi":
            self._model.save_model(str(path / "model.cbm"))
        else:
            for q, booster in self._models.items():
                booster.save_model(str(path / f"q{int(round(q * 100))}.cbm"))
        (path / "meta.json").write_text(json.dumps({
            "mode": self.mode,
            "feature_names": self._feature_names,
            "quantiles": list(self.quantiles),
            "params": self.params,
            "depth_by_quantile": {f"{q:.2f}": v for q, v in self.depth_by_quantile.items()},
            "l2_by_quantile": {f"{q:.2f}": v for q, v in self.l2_by_quantile.items()},
        }))

    @classmethod
    def load(cls, path: Path) -> "CatBoostQuantileForecaster":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        # Legacy checkpoints (pre-rewrite) don't have a "mode" key — default to
        # "multi" so they still load.
        mode = meta.get("mode", "multi")
        quantiles = tuple(meta["quantiles"])
        depth_by_q = None
        l2_by_q = None
        if "depth_by_quantile" in meta:
            depth_by_q = {float(k): int(v) for k, v in meta["depth_by_quantile"].items()}
        if "l2_by_quantile" in meta:
            l2_by_q = {float(k): float(v) for k, v in meta["l2_by_quantile"].items()}
        m = cls(
            quantiles=quantiles,
            params=meta.get("params"),
            depth_by_quantile=depth_by_q,
            l2_by_quantile=l2_by_q,
            mode=mode,
        )
        m._feature_names = meta["feature_names"]
        if mode == "multi":
            m._model = CatBoostRegressor()
            m._model.load_model(str(path / "model.cbm"))
        else:
            for q in quantiles:
                booster = CatBoostRegressor()
                booster.load_model(str(path / f"q{int(round(q * 100))}.cbm"))
                m._models[float(q)] = booster
        return m


# ---------------------------------------------------------------- LightGBM

class LightGBMQuantileForecaster(BaseQuantileForecaster):
    """One LightGBM booster per quantile, all sharing the same feature set."""

    def __init__(
        self,
        *,
        params: Optional[dict] = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int | None = 100,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        random_state: int = 42,
        **lgb_kwargs,
    ):
        super().__init__(name="lightgbm_q", quantiles=quantiles)
        self.params = {
            "objective": "quantile",
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "seed": random_state,
            "force_col_wise": True,
            "num_threads": -1,  # use all cores (xgboost-lightgbm skill: parallelize)
            **(params or {}),
            **lgb_kwargs,
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self._boosters: dict[float, lgb.Booster] = {}
        self._feature_names: list[str] | None = None
        self._cat_indices: list[int] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight=None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMQuantileForecaster":
        self._feature_names = select_feature_columns(X)
        # LightGBM natively supports float categorical features via the
        # ``categorical_feature`` index list (xgboost-lightgbm skill: "LightGBM
        # has native categorical feature support").
        self._cat_indices = detect_categorical_indices(X, self._feature_names)
        X_arr = to_array(X, self._feature_names)
        cat_arg = self._cat_indices or "auto"
        train_ds = lgb.Dataset(
            X_arr, label=y.to_numpy(), weight=sample_weight,
            categorical_feature=cat_arg, free_raw_data=False,
        )
        valid_sets, valid_names = [train_ds], ["train"]
        if X_val is not None and y_val is not None:
            X_val_arr = to_array(X_val, self._feature_names)
            valid_sets.append(lgb.Dataset(
                X_val_arr, label=y_val.to_numpy(), reference=train_ds,
                categorical_feature=cat_arg, free_raw_data=False,
            ))
            valid_names.append("val")
        for q in self.quantiles:
            params_q = {**self.params, "alpha": float(q)}
            callbacks = [lgb.log_evaluation(0)]
            if self.early_stopping_rounds and X_val is not None:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
            booster = lgb.train(
                params_q, train_ds, num_boost_round=self.num_boost_round,
                valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
            )
            self._boosters[float(q)] = booster
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._boosters:
            raise RuntimeError("must call fit() before predict_quantiles()")
        X_arr = to_array(X, self._feature_names)
        out = pd.DataFrame(index=X.index)
        for q, booster in self._boosters.items():
            out[f"q{int(round(q * 100))}"] = booster.predict(X_arr)
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for q, b in self._boosters.items():
            b.save_model(str(path / f"q{int(round(q * 100))}.txt"))
        (path / "meta.json").write_text(json.dumps({
            "feature_names": self._feature_names,
            "quantiles": list(self.quantiles),
            "params": self.params,
        }))

    @classmethod
    def load(cls, path: Path) -> "LightGBMQuantileForecaster":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        m = cls(quantiles=tuple(meta["quantiles"]), params=meta["params"])
        m._feature_names = meta["feature_names"]
        for q in meta["quantiles"]:
            m._boosters[float(q)] = lgb.Booster(model_file=str(path / f"q{int(round(q * 100))}.txt"))
        return m


# ---------------------------------------------------------------- Linear bases

