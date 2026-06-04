# Imputation Static Audit

_Generated against 24 columns with NaN in the raw source._

| Status | Column | NaN rate | mean(obs) | mean(imp) | std(obs) | std(imp) | KS p | Wasserstein | Physical issues | Reasons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 🔴 | `FR_to_DELU` | 56.4% | 839.71 | -546.68 | 1080.56 | 1955.89 | 0.00e+00 | 1386.39 | - | mean shift 1.28σ vs observed; imputed mean has opposite sign vs observed; KS p=0.00e+00 (distributions differ) |
| 🔴 | `gen_fossil_coal_gas` | 51.8% | 494.97 | 0.00 | 106.07 | 0.00 | 0.00e+00 | 494.97 | - | mean shift 4.67σ vs observed; variance collapse (0.00× of observed); KS p=0.00e+00 (distributions differ) |
| 🟢 | `carbon_ets` | 39.2% | 24.52 | None | 4.37 | None | None | None | - | - |
| 🔴 | `gen_nuclear` | 38.8% | 6240.29 | 10.47 | 2188.39 | 253.91 | 0.00e+00 | 6229.83 | - | mean shift 2.85σ vs observed; variance collapse (0.12× of observed); KS p=0.00e+00 (distributions differ) |
| 🟡 | `gen_fossil_oil` | 36.0% | 266.18 | 335.49 | 195.93 | 136.22 | 0.00e+00 | 88.70 | - | KS p=0.00e+00 (distributions differ) |
| 🟡 | `gen_other` | 31.4% | 303.85 | 204.07 | 104.26 | 80.66 | 0.00e+00 | 99.78 | - | mean shift 0.96σ vs observed; KS p=0.00e+00 (distributions differ) |
| 🔴 | `DELU_to_CH` | 21.5% | 1842.26 | -268.00 | 3069.97 | 815.54 | 0.00e+00 | 2110.26 | - | mean shift 0.69σ vs observed; imputed mean has opposite sign vs observed; variance collapse (0.27× of observed); KS p=0.00e+00 (distributions differ) |
| 🔴 | `DELU_to_FR` | 15.1% | 2721.46 | -203.63 | 3320.91 | 241.50 | 0.00e+00 | 2925.09 | - | mean shift 0.88σ vs observed; imputed mean has opposite sign vs observed; variance collapse (0.07× of observed); KS p=0.00e+00 (distributions differ) |
| 🟡 | `gen_waste` | 6.6% | 727.91 | 609.77 | 129.50 | 173.18 | 0.00e+00 | 118.15 | - | mean shift 0.91σ vs observed; KS p=0.00e+00 (distributions differ) |
| 🔴 | `CH_to_DELU` | 4.2% | 3539.19 | -129.05 | 4109.04 | 208.68 | 0.00e+00 | 3668.24 | - | mean shift 0.89σ vs observed; imputed mean has opposite sign vs observed; variance collapse (0.05× of observed); KS p=0.00e+00 (distributions differ) |
| 🔴 | `gen_solar` | 3.6% | 6440.44 | 0.96 | 9858.44 | 11.83 | 0.00e+00 | 6439.48 | - | mean shift 0.65σ vs observed; variance collapse (0.00× of observed); KS p=0.00e+00 (distributions differ) |
| 🟢 | `gen_geothermal` | 2.9% | 23.16 | 23.32 | 5.72 | 5.51 | 7.21e-02 | 0.24 | - | - |
| 🟢 | `gen_other_renewable` | 1.0% | 125.11 | 124.57 | 37.66 | 37.81 | 2.59e-01 | 1.83 | - | - |
| 🟢 | `gen_hydro_ror` | 0.1% | 1554.89 | 1530.08 | 310.57 | 356.37 | 4.30e-01 | 52.17 | - | - |
| 🔴 | `gen_hydro_reservoir` | 0.1% | 180.31 | 63.08 | 183.91 | 40.88 | 7.80e-12 | 120.71 | - | mean shift 0.64σ vs observed; variance collapse (0.22× of observed); KS p=7.80e-12 (distributions differ) |
| 🔴 | `gen_wind_offshore` | 0.1% | 2857.31 | 461.71 | 1925.18 | 1097.43 | 4.03e-26 | 2395.59 | - | mean shift 1.24σ vs observed; KS p=4.03e-26 (distributions differ) |
| 🔴 | `ttf_gas` | 0.0% | 45.11 | 22.47 | 44.68 | 0.01 | None | None | - | mean shift 0.51σ vs observed; variance collapse (0.00× of observed) |
| 🔴 | `DELU_to_AT` | 0.0% | 1121.94 | -12.36 | 2360.27 | 35.65 | None | None | - | imputed mean has opposite sign vs observed; variance collapse (0.02× of observed) |
| 🟢 | `gen_biomass` | 0.0% | 4318.24 | 4296.67 | 340.84 | 343.50 | None | None | - | - |
| 🟡 | `gen_fossil_hard_coal` | 0.0% | 4772.90 | 2584.67 | 3353.24 | 1763.84 | None | None | - | mean shift 0.65σ vs observed |
| 🟢 | `gen_fossil_gas` | 0.0% | 6008.50 | 4262.74 | 3642.78 | 2255.94 | None | None | - | - |
| 🟡 | `gen_fossil_lignite` | 0.0% | 9836.06 | 11740.33 | 3650.40 | 1214.41 | None | None | - | mean shift 0.52σ vs observed |
| 🔴 | `gen_wind_onshore` | 0.0% | 11921.85 | 3422.26 | 9476.57 | 0.00 | None | None | - | mean shift 0.90σ vs observed; variance collapse (0.00× of observed) |
| 🔴 | `AT_to_DELU` | 0.0% | 3539.02 | 1766.69 | 3454.20 | 0.00 | None | None | - | mean shift 0.51σ vs observed; variance collapse (0.00× of observed) |

**Summary:** 🟢 6 | 🟡 5 | 🔴 13