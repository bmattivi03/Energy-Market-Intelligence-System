# Module B — Day-ahead price forecasting (DE-LU)

Probabilistic forecaster for the Germany–Luxembourg day-ahead electricity
market. Produces hourly quantile predictions (`q10`, `q50`, `q90`) over a
24-hour horizon, designed to feed Module C's CVaR-aware battery-dispatch RL
agent.

## Status (locked test set, 2025 Q1 — SSHB-imputed splits)

| model         | MAE h1–6 | pinball | coverage_80 |
|---------------|---------:|--------:|------------:|
| naive         | 47.62    | 15.70   | 0.74        |
| sn168         | 43.34    | 14.58   | 0.69        |
| LightGBM+CQR  | 24.75    | 8.23    | 0.83        |
| **CatBoost+CQR**  | **23.29** | **7.93**    | **0.82**        |

CatBoost+CQR beats seasonal-naive-168h by ~46% on MAE/pinball with near-nominal
80% coverage; Diebold-Mariano confirms the gap is significant. Numbers are on the
new SSHB-imputed splits (prior imputation gave 23.15); model roster reduced to
CatBoost + LightGBM + naive baselines (LEAR/Ridge/ElasticNet removed). Full
per-horizon results in `reports/module_b_final_leaderboard.md`.

## Layout

Three files, one responsibility each:

| File | Contents |
|------|----------|
| `features.py` | Feature transforms (calendar, price lags + rolling stats, fundamentals, spike flags, regime flags, weather lags/aggregates), the `REGISTRY` of bundles with dependency resolution, and `prepare_supervised` which builds the `(origin_ts, horizon_h) → (X, y)` layout used by every model. |
| `models.py` | `BaseQuantileForecaster` ABC plus four forecasters: `NaiveForecaster`, `SeasonalNaiveForecaster`, `CatBoostQuantileForecaster` (production), `LightGBMQuantileForecaster` (fallback). Weak linear models (LEAR/Ridge/ElasticNet) were removed in the 2026-06 simplification — they were 30–67% worse than CatBoost. |
| `evaluation.py` | Point metrics (`mae`, `rmse`, `smape`, `spike_mae`, `directional_accuracy`), probabilistic metrics (`pinball_loss`, `multi_pinball_loss`, `coverage`, `winkler_score`), 9 segmentation slicers (`SEGMENT_FNS`), `diebold_mariano` significance test, `bootstrap_ci`, and the calibration wrappers `ConformalQuantileRegressor` (CQR) and `AdaptiveConformalCalibrator` (ACP). |

All experiments live as notebooks under `notebooks/module_b/`; the package
only provides reusable building blocks.

## Models

### Baselines (B2)
- `NaiveForecaster` — `ŷ_{T+h} = price_{T-1h}` for all `h`.
- `SeasonalNaiveForecaster(season_hours=168)` — `ŷ_{T+h} = price_{T+h-168}`.
- Quantile bands come from empirical residual bootstrap (`ResidualQuantileEstimator`).

### Classical (B3, with rolling-origin CV)
- **`CatBoostQuantileForecaster`** — *per-quantile* by default since the
  2026-05 rewrite. Fits three independent boosters (`Quantile:alpha=0.1`,
  `0.5`, `0.9`) with `depth=8` for the tails (q10, q90) and `depth=6` for
  the median, `l2_leaf_reg=1` for tails and `3` for median. The previous
  `MultiQuantile` mode (under-fit upper tails — see
  `reports/module_b_catboost_calibration_diagnosis.md`) is kept under
  `mode="multi"` for legacy checkpoint compatibility.
- `LightGBMQuantileForecaster` — one booster per quantile, native handling
  of binary categorical features via `detect_categorical_indices`. Kept as a
  validated fallback (≈8% behind CatBoost on the locked test).

### Calibration (B5)
- `ConformalQuantileRegressor` — wraps any base, learns one additive `delta`
  on a held-out calibration set so coverage_80 hits the 80% nominal target
  on test (Romano-Patterson-Candès 2019). Uses the
  `⌈(n+1)(1−α)⌉/n` finite-sample quantile.
- `AdaptiveConformalCalibrator` — online Gibbs-Candès (2021) recursion that
  adapts to distribution shift; available but not currently in the
  production path.

## Data flow

```
data/splits/{train,val,test}.parquet      (loaded via src/data/loaders.load_split)
        │
        ▼
build_features(bundles=("calendar", "lags", "fundamentals",
                        "spike", "regime", "weather"))           features.py
        │
        ▼
prepare_supervised(horizons=range(1, 25), past_cols=…, future_cols=…)
        │  ── one row per (origin_ts, horizon_h) pair
        │  ── target = price.shift(-h), future_cols at T+h, past_cols at T
        ▼
BaseQuantileForecaster.fit / predict_quantiles                   models.py
        │
        ▼
ConformalQuantileRegressor.calibrate / predict_quantiles         evaluation.py
        │
        ▼
mae, multi_pinball_loss, coverage, segment_metrics, diebold_mariano
```

## Temporal hygiene

- Splits: train 2019-01 → 2023-12 (43 824 h), val 2024 (8 784 h), test
  2025-Q1 (~2 160 h). Locked in `src/data/splits.py`.
- Rolling-origin CV (B3) uses a 24-hour gap between train and val windows
  to prevent lag-24 leakage.
- All features are causal: rolling stats use `.shift(1)`, spike-threshold
  quantiles use `expanding(min_periods=720).quantile(0.9)` (strictly
  cumulative in pandas), and the supervised layout has past_cols at `T`
  and future_cols at `T+h` only.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `B1_features.ipynb` | Build + visualise calendar / lag / fundamentals / spike / regime / weather bundles. |
| `B2_baselines.ipynb` | Fit naive / seasonal-24h / seasonal-168h baselines on 2024 val. |
| `B3_classical.ipynb` | 3-fold rolling-origin CV for CatBoost / LightGBM. |
| `B5_calibration.ipynb` | Fit + CQR-calibrate each base model; per-horizon coverage; sample-week plot. |
| `B6_final_eval.ipynb` | Locked 2025 Q1 test set: leaderboard, Diebold-Mariano, segment breakdowns, sample-week plot. |

## Reproducibility

- `pytest tests/module_b/` runs 53 unit + property tests — all pass.
- Each forecaster takes a `random_state` kwarg that seeds the underlying
  library RNG (CatBoost / LightGBM / sklearn).

## Related reports

- `reports/module_b_catboost_calibration_diagnosis.md` — why the original
  CatBoost `MultiQuantile` mode under-fit the upper tail (35.5% of `y` above
  `q90` on val), and the post-fix results after the per-quantile rewrite
  (6.8% above `q90`, base coverage 0.49 → 0.75).
- `reports/module_b_full_vs_light_preset.md` — why CatBoost's "full"
  preset performed worse than "light" on the harder folds (early-stopping
  noise tracking on 90-day windows).
- `reports/module_b_catboost_optuna.md` — 10-trial Optuna sweep result:
  defaults are within 2% of optimal; no lift available from hyperparameter
  tuning alone.
