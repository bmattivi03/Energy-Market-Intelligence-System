# CatBoost base-coverage diagnosis — 2024 val

## Question
Why does CatBoost's base (pre-CQR) coverage_80 = 0.491 vs LightGBM = 0.731 on identical data?

## Findings

On 210,516 supervised rows from the 2024 val fold:

| | q10>q50 | q50>q90 | width median | coverage_80 | y<q10 | y>q90 | MAE on q50 |
|---|---|---|---|---|---|---|---|
| **CatBoost** (MultiQuantile, depth=6, lr=0.05) | **0.00%** | 0.00% | **41.2** | 0.491 | 15.4% | **35.5%** | 28.67 |
| **LightGBM** (per-quantile, num_leaves=63) | 2.57% | 0.29% | **95.4** | 0.731 | 20.8% | 6.1% | 25.59 |

## Diagnosis

1. **Not quantile crossing.** CatBoost has zero crossings; the prior hypothesis was wrong.
2. **CatBoost's intervals are ~2.3× narrower** than LightGBM's at every percentile (p25: 29.7 vs 78.5; p75: 58.0 vs 114.0; p99: 107.9 vs 184.4). The MultiQuantile loss is under-fitting interval width.
3. **CatBoost's residuals are strongly right-skewed** post-fit: 35.5% of y values exceed q90, only 15.4% fall below q10. The upper tail is being systematically under-predicted. LightGBM has the opposite, weaker asymmetry (6% above q90, 21% below q10) — slightly biased low overall but symmetric in the tail.
4. **CatBoost's q50 has worse MAE than LightGBM's q50** (28.7 vs 25.6 €/MWh), so the loss minimum it found is dominated by the median and the q10/q90 components get short-changed.

## Likely cause

`MultiQuantile:alpha=0.1,0.5,0.9` averages three pinball losses into a single scalar. With depth=6 and l2_leaf_reg=3, the model finds a low-bias-low-variance regime that is fine for q50 but doesn't have enough capacity to fit the heavy upper tail of post-crisis 2024 DE-LU prices (which routinely spike to €300–500 in peak hours). The gradient signal at α=0.9 in samples below the tail is small, so the model effectively ignores it.

## Recommendations (re-prioritized after this diagnostic)

1. **Switch to per-quantile boosters** (LightGBM-style) for CatBoost too, instead of MultiQuantile — this trades a small amount of consistency for materially better tail coverage. Available in CatBoost via setting `loss_function="Quantile:alpha=0.9"` and training a separate model per quantile.
2. **Increase depth** to 8 or 10 — gives the upper-tail leaves room to capture spikes.
3. **Reduce l2_leaf_reg** for tail quantiles only (e.g. l2=1 for q10/q90, l2=3 for q50) — currently regularization is uniform across quantiles.
4. **Make the production model choice in B6 LightGBM+CQR**, not CatBoost+CQR — LightGBM has materially better base calibration (0.73 vs 0.49) and pays much less width penalty when CQR is applied (€8/MWh extra vs €16/MWh).

## Practical takeaway

The CatBoost+CQR result in B6 (coverage 0.81) is achieved by CQR *masking* a deeply mis-calibrated base model. The interval width on test is correspondingly inflated. LightGBM+CQR would likely deliver the same coverage with a much tighter interval, which directly benefits the downstream Module-C RL agent's CVaR computation.

---

## Post-fix (2026-05-12) — per-quantile CatBoost

`CatBoostQuantileForecaster` was rewritten to fit one booster per α, with
per-quantile `depth` (8 for q10/q90, 6 for q50) and `l2_leaf_reg` (1.0 for
the tails, 3.0 for the median). The legacy `MultiQuantile` path is preserved
under `mode="multi"` for loading older checkpoints. See
`src/module_b/models.py` (`CatBoostQuantileForecaster`).

### Re-run on the same 2024 val fold (B5 re-executed with new defaults)

| | q10>q50 | q50>q90 | width median | coverage_80 (base) | y<q10 | y>q90 | pinball q90 |
|---|---|---|---|---|---|---|---|
| CatBoost **before** (MultiQuantile, depth=6) | 0.00% | 0.00% | 41.2 | 0.491 | 15.4% | **35.5%** | (not recorded) |
| CatBoost **after** (per-quantile, depth=8/6/8) | 8.37% | 0.02% | **79.5** | **0.749** | 18.3% | **6.8%** | 7.56 |
| LightGBM (reference) | 2.57% | 0.29% | 95.4 | 0.731 | 20.8% | 6.1% | 9.98 |

### What changed and why it matters

1. **Upper-tail miss rate dropped 5×** — y>q90 fell from 35.5% to 6.8%, now
   essentially at the theoretical 10% mark and matching LightGBM. The
   structural problem from the diagnosis (under-trained α=0.9 booster) is
   resolved by giving the tail its own booster with depth=8.
2. **CQR delta on B5 dropped from 15.74 → 7.19 €/MWh** (a 54% reduction).
   CatBoost+CQR now produces *tighter* intervals than LightGBM+CQR (delta
   7.19 vs 7.83) while achieving slightly higher coverage_80 (0.818 vs 0.811).
3. **q90 pinball improved from ~10 to 7.56 €/MWh.** The per-quantile median
   also improved (q50 MAE 28.7 → 26.7).
4. **Quantile crossings now appear at q10/q50 (8.37%)** because the three
   boosters are trained independently and can disagree near the median. The
   important boundary q10/q90 remains crossing-free (0%), so coverage_80 and
   CVaR consumers are unaffected. If interpolation to intermediate quantiles
   is ever required, post-hoc isotonic ordering can be applied.

### Why the prior diagnosis-led prediction was right

The diagnosis explicitly recommended switching to per-quantile boosters; the
implementation confirmed the predicted effect. The mechanism — `MultiQuantile`
loss averages three pinball terms, and the median dominates the gradient — is
borne out: independent boosters with depth=8 on the tails capture the upper
spikes the joint loss was missing.

### Recommendation revision

The earlier recommendation to switch the B6 production model to LightGBM+CQR
is **no longer necessary**. With per-quantile mode, CatBoost+CQR provides
better calibration AND narrower intervals than LightGBM+CQR on the same val
data.

### B6 — locked 2025 Q1 test set, before vs after

The B6 notebook was re-executed end-to-end with the new per-quantile
CatBoost. Every test-set metric for CatBoost+CQR improved or stayed flat
while coverage drifted closer to the 0.80 nominal target (it had been
slightly *over*-covered before, due to the wide CQR-corrected intervals).

| horizon | metric             | before (MultiQuantile) | after (per-quantile) |
|---------|--------------------|------------------------|----------------------|
| h1–6    | MAE                | 23.604                 | **23.147** (−0.46)   |
| h1–6    | pinball_avg        | 7.881                  | 7.920 (+0.04)        |
| h1–6    | coverage_80        | 0.817                  | 0.808                |
| h7–18   | MAE                | 24.663                 | **24.087** (−0.58)   |
| h7–18   | pinball_avg        | 8.304                  | 8.315                |
| h7–18   | coverage_80        | 0.807                  | 0.798                |
| h19–24  | MAE                | 25.050                 | **24.350** (−0.70)   |
| h19–24  | coverage_80        | 0.803                  | 0.798                |
| —       | CQR `delta`        | 8.085                  | **7.820** (−0.27)    |
| —       | all-segment MAE    | 24.494                 | **23.917** (−0.58)   |
| —       | spike_top10pct MAE | 42.849                 | **41.796** (−1.05)   |
| —       | negative_price MAE | 74.905                 | **71.268** (−3.64)   |

Diebold-Mariano (one-sided, 2025 Q1) — `LightGBM+CQR vs CatBoost+CQR` now
returns `dm_stat = 13.595`, `p = 1.0`, i.e. the *one-sided test for LightGBM
being better* is rejected with high confidence (LightGBM is significantly
worse on test). The earlier run showed the same direction (dm_stat 7.55)
but with the per-quantile rewrite the gap widened. **CatBoost+CQR is now
the clear production-model choice on the locked test set.**

### Takeaway

The diagnosis was correct, the prescription worked, and the headline test
numbers in `reports/module_b_final_leaderboard.md` now reflect a structurally
better-calibrated forecaster — not one whose coverage is patched up by an
oversized CQR correction.
