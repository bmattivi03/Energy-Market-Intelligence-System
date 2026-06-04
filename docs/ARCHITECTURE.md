# Architecture

This document is the engineering reference for the Energy Market Intelligence
System. It explains how data flows through the pipeline, the conventions every
module follows, and the reproducibility guarantees. For results and the project
narrative, see the top-level `README.md` and `final_report.pdf`.

## Organising principle: uncertainty propagation

The system is built around one idea: predictive uncertainty is made explicit and
propagated forward, rather than collapsed to point estimates. Every forecasting
module emits calibrated quantiles (q10, q50, q90), and the downstream consumer
receives those quantiles as input rather than a single number.

```
data/raw/  ->  preprocessing/      ->  Module A (load)    ->\
               Glocal-IB SSHB imputer  multi-scale LSTM      \   Module C (RL)
               + physical constraints  -> q10/q50/q90         ->  PPO / SAC + CVaR
                                                              /   -> battery dispatch
                                        Module B (price)     /
                                        per-quantile CatBoost
                                        -> q10/q50/q90
```

Module B's price quantiles are part of Module C's observation. Module A's load
quantiles are an optional, ablation-gated input to Module C. The A-to-B link was
tested and found not to help (see the cascade analysis in the report), so the
production price model does not consume load quantiles.

## Data layer (`src/data/`)

Every module reads data through `src.data.loaders.load_split("train"|"val"|"test")`.
No module reads parquet files directly. This keeps column schemas, split bounds,
and cross-validation iterators in a single source of truth.

- `schemas.py`: `SPLIT_BOUNDS` (hard-coded datetime ranges) and the
  structural-zero indicator specification.
- `splits.py`: `HORIZON = 24`, `LOOKBACK = 168`, plus `rolling_origin_folds`
  and `expanding_window_folds`. Both CV iterators insert a 24-hour `gap` between
  train and validation so that a lag-24 feature cannot leak the target.
- `loaders.py`: `load_split`, `load_train`, `load_val`, `load_test`.

The train / validation / test split is fixed and must not be changed:

| Split | Range | Hours |
|-------|-------|-------|
| train | 2019-01 to 2023-12 | 43,824 |
| val   | 2024 | 8,784 |
| test  | 2025-01 to 2025-03 (locked) | 2,160 |

The locked test quarter is touched only by `notebooks/module_b/B6_final_eval.ipynb`
and the module evaluation entry points. Touching it anywhere else is overfitting.

## Raw sources

Six raw parquet files under `data/raw/`:

| File | Source | Content |
|------|--------|---------|
| `entsoe_prices_2019_2025.parquet` | ENTSO-E A44 | Hourly day-ahead prices (EUR/MWh) |
| `entsoe_load_2019_2025.parquet` | ENTSO-E A65 | Hourly actual load (MW) |
| `entsoe_generation_2019_2025.parquet` | ENTSO-E A75 | Hourly generation by production type (17 sources) |
| `entsoe_crossborder_2019_2025.parquet` | ENTSO-E A11 | Cross-border flows plus derived net imports (DE to FR/AT/CH) |
| `weather_2019_2025.parquet` | Open-Meteo | Hourly weather for 5 German cities |
| `fuels_2019_2025.parquet` | yfinance | Daily TTF gas and Carbon ETS |

Ingestion notes that matter operationally live in the ingestion modules:

- Two EIC codes are both required: `10Y1001A1001A83F` (physical: load,
  generation, flows) and `10Y1001A1001A82H` (market: prices A44).
- A75 (generation per source) requires 90-day request chunks.
- B03 (coal/gas generation) returns empty for Germany 2019 to 2022; this is a
  data-availability fact on the platform, not a rate-limit, and is handled as a
  structural zero downstream.
- Ingestion is resumable via `data/raw/.checkpoint.json`.

## Imputation (`src/preprocessing/` + `GlocalIB/`)

The raw panel has 47 columns; 24 of them contain missing values, concentrated in
cross-border flows, fossil generation, and the carbon series. The imputation
stage reconstructs a complete panel.

Pipeline (`preprocessing/impute.py`):

1. Load `emis_raw.parquet`, apply domain fixes (zero-fill coal/gas for the
   2019 to 2022 no-data window; zero-fill nuclear after the 2023-04-15 shutdown).
2. Fit a `NaNScaler` on train rows only.
3. Build 168-step windows with stride 24 (validation receives a 20% MCAR mask
   for honest evaluation).
4. Train `SAITS_MY` (the improved Glocal-IB / SAITS variant in `GlocalIB/`).
5. Sliding-window inference with overlap averaging; observed values are never
   overwritten.
6. Inverse-scale, recompute derived net-import columns, and save
   `emis_imputed.parquet` plus `emis_mask.parquet`.

The vendored `GlocalIB/` fork exposes two project additions as mod-flags:

- `mod_e` (Variational Information Bottleneck): a training-time regulariser.
- `mod_p` (Self-Supervised Held-out Blend, SSHB): an inference-time blend of the
  model output with a local smoother, where the blend weight is chosen per
  feature by honest cross-validation on held-out observed cells. SSHB requires
  no retraining, so the A/B comparison is "train once, run inference twice".

Two post-imputation fixes are applied in `build_splits.py`:

1. `carbon_ets` is forward and backward filled to cover exchange holidays.
2. `gen_fossil_coal_gas` values synthesised for the 2019 to 2022 no-data window
   are replaced with 0, and a companion `gen_fossil_coal_gas_structural_zero`
   column marks those rows. Downstream models treat `structural_zero == 1` as
   missing rather than as real signal.

After imputation and fixes the model-ready panel (`emis_imputed.parquet`) has
52 columns.

## Module A: load forecasting (`src/module_a/`)

A multi-scale LSTM: a short branch over a 48-hour window for intraday dynamics, a
long branch over a 168-hour window for weekly seasonality, and a shared MLP head
that emits q10, q50, q90 for the next 24 hours via the pinball loss. The raw
network is sharp but under-dispersed, so a Conformalized Quantile Regression
(CQR) step widens the interval to restore near-nominal 80% coverage.

## Module B: price forecasting (`src/module_b/`)

Flat three-file layout: `models.py`, `features.py`, `evaluation.py`.

- `CatBoostQuantileForecaster` defaults to `mode="per_quantile"`: three
  independent boosters, one per quantile, with asymmetric depth and L2 (8/6/8 and
  1/3/1 for q10/q50/q90). The legacy joint MultiQuantile mode is retained only to
  load old checkpoints. The per-quantile design fixed an upper-tail
  miscalibration (fraction of outcomes above q90 fell from 35.5% to 6.8%).
- All features are causal. Rolling statistics use `.shift(1)`; the spike-threshold
  quantile uses a strictly cumulative `expanding(min_periods=720).quantile(0.9)`;
  `prepare_supervised` shifts past columns at T, future columns at T+h, and the
  target at T+h.
- `ConformalQuantileRegressor` uses the Barber finite-sample quantile
  `ceil((n+1)(1-alpha)) / n`. `AdaptiveConformalCalibrator` (Gibbs-Candes online
  recursion) is available but not in the production path.

Production model on the locked test set is CatBoost + CQR.

## Module C: battery dispatch (`src/module_c/`)

A virtual battery operated as a Markov decision process: 100 MWh energy capacity,
50 MW power, 90% round-trip efficiency, 24-hour episodes. The action is a
continuous scalar in [-1, 1] mapped to [-50, +50] MW. The observation is
78-dimensional: state of charge, hour and day-of-week cyclicals, the standardised
lag-1 price, and the full Module B forecast (q10, q50, q90 across horizons).

The reward is per-step profit minus a CVaR penalty,
`r = profit - 0.1 * max(0, -CVaR_0.05)`, computed over a rolling 168-hour window.
Two agents are trained for 200k steps each: PPO (on-policy) and SAC (off-policy).
SAC is the production agent. An observation ablation (raw / B / B+A) quantifies
the marginal value of each upstream forecast.

## Orchestration and reproducibility

- `Makefile` and `scripts/run_pipeline.py` drive the pipeline end to end with
  skip-if-exists caching (`--from`, `--only`, `--skip-*`, `--force`).
- A single seed (default 42) controls NumPy, Python, and PyTorch. Device
  selection prefers CUDA, then Apple MPS, then CPU. The imputation pipeline has a
  CPU fallback and the RL training is pinned to CPU because the Apple Silicon GPU
  backend segfaults under Stable-Baselines3.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` when running the imputation pipeline on
  Apple Silicon.

## Testing

The suite lives under `tests/` and assumes `src/` is importable;
`tests/conftest.py` inserts it automatically, so `pytest tests/` works from the
repository root. `tests/module_b/conftest.py` provides a synthetic `small_df`
fixture used by the feature and model tests, including property tests for the CQR
marginal-coverage guarantee and the Barber-quantile formula.
