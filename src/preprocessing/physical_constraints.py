"""Post-hoc physical-bound enforcement for the imputed dataset.

Each function takes a DataFrame, returns a *new* DataFrame with constraints
applied. None of these functions modify their input. Designed to be composed
in ``apply_all`` after any imputer (Glocal-IB, KNN, linear interpolation, …).

Constants describing the physical model live alongside the functions so the
audit harness (``imputation_eval.py``) can reuse them - see
``GENERATION_NONNEG_COLS`` and ``CROSS_BORDER_PAIRS`` re-exported from
``imputation_eval``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.imputation_eval import (
    CROSS_BORDER_PAIRS,
    GENERATION_NONNEG_COLS,
    compute_sun_elevation,
)

# KEUA ETF (carbon ETS proxy) inception date.
CARBON_ETS_INCEPTION = pd.Timestamp("2021-09-29", tz="UTC")
CARBON_ETS_PRE_LAUNCH_INDICATOR = "carbon_ets_pre_launch_indicator"
STRUCTURAL_ZERO_INDICATOR = "gen_fossil_coal_gas_structural_zero"

# Solar elevation threshold (degrees below horizon) defining "full night".
# −6° is civil twilight - the conservative threshold below which any non-zero
# PV generation is physically impossible.
SOLAR_NIGHT_ELEVATION_THRESHOLD = -6.0


def clip_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Clip every non-negative generation column to ``[0, +inf)``.

    ``gen_hydro_pumped`` is intentionally excluded - pumped storage consumption
    is naturally negative.
    """
    out = df.copy()
    for col in GENERATION_NONNEG_COLS:
        if col in out.columns:
            out[col] = out[col].clip(lower=0.0)
    return out


def enforce_solar_night_zero(
    df: pd.DataFrame,
    *,
    lat: float = 51.0,
    lon: float = 10.0,
    column: str = "gen_solar",
    night_threshold: float = SOLAR_NIGHT_ELEVATION_THRESHOLD,
    twilight_threshold: float = 0.0,
) -> pd.DataFrame:
    """Force ``gen_solar`` to 0 below civil twilight; taper linearly through twilight.

    Below ``night_threshold`` (default −6°): hard zero.
    Between ``night_threshold`` and ``twilight_threshold``: linear ramp (× 0 → 1).
    Above ``twilight_threshold``: untouched.

    The ramp avoids discontinuities that would distort downstream gradient
    features (e.g. solar ramp at sunrise).
    """
    if column not in df.columns:
        return df.copy()
    out = df.copy()
    elev = compute_sun_elevation(df.index, lat=lat, lon=lon)
    weight = np.clip(
        (elev - night_threshold) / (twilight_threshold - night_threshold),
        0.0,
        1.0,
    )
    out[column] = out[column] * weight
    return out


def enforce_flow_sign_coherence(df: pd.DataFrame) -> pd.DataFrame:
    """For each border (FR/AT/CH), prevent both directional flow columns from
    being simultaneously positive. Resolves by keeping the larger and zeroing
    the smaller (preserving the net-flow sign and magnitude).
    """
    out = df.copy()
    for a, b in CROSS_BORDER_PAIRS:
        if a not in out.columns or b not in out.columns:
            continue
        a_vals = out[a].fillna(0.0)
        b_vals = out[b].fillna(0.0)
        both_pos = (a_vals > 0) & (b_vals > 0)
        # Keep larger, zero smaller
        keep_a = both_pos & (a_vals >= b_vals)
        keep_b = both_pos & (a_vals < b_vals)
        out.loc[keep_a, b] = 0.0
        out.loc[keep_b, a] = 0.0
    return out


def restore_carbon_ets_pre_launch(
    df: pd.DataFrame,
    inception: pd.Timestamp = CARBON_ETS_INCEPTION,
) -> pd.DataFrame:
    """Set ``carbon_ets`` to NaN for all rows before the KEUA ETF inception
    (2021-09-29) and add an indicator column.

    The current notebook fix (ffill+bfill) fabricates ~24k rows of carbon-price
    signal by extrapolating the very first observed tick backwards through 2019.
    This restores the principled NaN + indicator encoding so downstream models
    can choose: (a) drop pre-launch rows, (b) include indicator as a feature,
    (c) use a separate pre-launch baseline.
    """
    if "carbon_ets" not in df.columns:
        return df.copy()
    out = df.copy()
    pre_launch = out.index < inception
    out.loc[pre_launch, "carbon_ets"] = np.nan
    out[CARBON_ETS_PRE_LAUNCH_INDICATOR] = pre_launch.astype(np.float32)
    return out


def add_structural_zero_indicator(
    imputed_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    column: str = "gen_fossil_coal_gas",
    indicator_name: str = STRUCTURAL_ZERO_INDICATOR,
) -> pd.DataFrame:
    """Propagate the structural-zero indicator into the imputed parquet itself.

    The original ``apply_domain_fixes`` zero-fills ``gen_fossil_coal_gas`` for
    all rows (because ENTSO-E genuinely returns no records for Germany 2019-22).
    Notebook 03 then adds an indicator column to the *splits* but the imputed
    parquet itself loses this information. Anything reading the parquet directly
    (e.g. an ad-hoc analysis script) cannot distinguish a genuine 0 from a
    structural NaN. This function fixes that by adding the indicator at the
    parquet level.
    """
    out = imputed_df.copy()
    if column not in raw_df.columns:
        out[indicator_name] = 0.0
        return out
    out[indicator_name] = raw_df[column].isna().astype(np.float32)
    return out


# ---- composition --------------------------------------------------------------

@dataclass
class ConstraintConfig:
    """Toggle which constraints to apply in ``apply_all``."""

    clip_generation: bool = True
    enforce_solar_night: bool = True
    enforce_flow_coherence: bool = True
    restore_carbon_ets: bool = True
    add_structural_zero: bool = True


def apply_all(
    imputed_df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    config: ConstraintConfig | None = None,
) -> pd.DataFrame:
    """Apply every enabled physical constraint, in dependency order.

    ``raw_df`` is required only if ``add_structural_zero`` is True (since the
    indicator is derived from the raw NaN pattern, not the imputed values).
    """
    cfg = config or ConstraintConfig()
    out = imputed_df
    if cfg.clip_generation:
        out = clip_generation(out)
    if cfg.enforce_solar_night:
        out = enforce_solar_night_zero(out)
    if cfg.enforce_flow_coherence:
        out = enforce_flow_sign_coherence(out)
    if cfg.restore_carbon_ets:
        out = restore_carbon_ets_pre_launch(out)
    if cfg.add_structural_zero:
        if raw_df is None:
            raise ValueError("add_structural_zero requires raw_df")
        out = add_structural_zero_indicator(out, raw_df)
    return out


# ---- CLI ----------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply physical constraints to an imputed parquet (post-hoc)."
    )
    parser.add_argument("--imputed", type=Path, default=Path("data/processed/emis_imputed.parquet"))
    parser.add_argument("--raw", type=Path, default=Path("data/processed/emis_raw.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/emis_imputed_clipped.parquet"))
    args = parser.parse_args()

    imputed = pd.read_parquet(args.imputed)
    raw = pd.read_parquet(args.raw)
    fixed = apply_all(imputed, raw_df=raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_parquet(args.out)
    print(f"Wrote {args.out}  ({fixed.shape[0]:,} rows × {fixed.shape[1]} columns)")


if __name__ == "__main__":
    _cli()
