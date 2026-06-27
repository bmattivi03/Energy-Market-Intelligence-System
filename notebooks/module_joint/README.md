# Module Joint: notebooks and run guide

`module_joint` is a single multi-task deep learning model that forecasts DE-LU
load and day-ahead price jointly (q10/q50/q90 over h1-24) from a shared encoder.
It is a pure deep-learning deliverable. Module A (LSTM) and Module B (CatBoost)
are imported only as reported baselines and feature/calibration sources; they are
never modified.

Design spec: `docs/superpowers/specs/2026-06-27-joint-multitask-load-price-deep-learning-design.md`
Implementation plan: `docs/superpowers/plans/2026-06-27-module-joint-mtl-implementation.md`

## Environment

```bash
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1     # Apple Silicon
# EMIS_FORCE_CPU=1 to force CPU
```

## Run order (phases)

The notebooks are thin drivers over `src/module_joint`. Run them in order. The
locked test split (`load_split("test")`) is touched ONLY in J6.

| Notebook | Phase | What it does |
|----------|-------|--------------|
| `J1_data_windows.ipynb` | 0 | Build and sanity-check windows (past, known-future + Module A load quantiles, targets, causal spike target) |
| `J2_backbone_bench.ipynb` | 1 | Single-task bench on 2024 val: DLinear / sn168 floors + TiDE + N-HiTSx. Pick the winner |
| `J3_joint_mtl.ipynb` | 1 | Train the joint TiDE MTL model; per-target STL-vs-MTL ship/no-ship gate |
| `J4_ablations.ipynb` | 2 | Pillar 1 load->price ladder (run first), spike head/asinh, aux heads, loss weighting, RevIN |
| `J5_ensemble.ipynb` | 3 | >=4-seed ensemble + conformal re-fit on the ensemble |
| `J6_final_eval.ipynb` | 3 | Locked 2025-Q1 leaderboard vs Module A/B + sn168, Diebold-Mariano, DM power, PIT/coverage |

## CLI (single training run)

```bash
PYTHONPATH=src python -m module_joint.train --backbone tide --epochs 80 --seed 42 \
  --out checkpoints/module_joint/model
```

## Notes

- All models in the package emit the same contract: `predict_quantiles` returns
  `dict[target] -> DataFrame` indexed by `(origin_ts, horizon_h)` with columns
  `q10, q50, q90` (load in MW, price in EUR/MWh), non-crossing by construction.
- Validation/test prediction prepends 168h of lookback context
  (`pd.concat([train.tail(168), val])`) and restricts emitted origins to the
  split via `restrict_to=split.index`, so lookback may use prior-split data while
  targets stay inside the split.
- Conformal calibration (`calibrate.conformalize`) adjusts q10/q90 endpoints only;
  q50 is never touched, to protect the MAE-scored point forecast.
- Tests: `PYTHONPATH=src pytest tests/module_joint/` (add `-m slow` for the
  end-to-end real-data smoke).
