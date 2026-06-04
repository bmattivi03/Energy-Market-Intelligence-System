"""Imputation validation harness.

Produces quantitative quality scores for an imputed dataset against the raw
(NaN-bearing) source. Two evaluation modes:

1. ``static_audit`` - no imputer required. Computes per-column distribution
   distance, autocorrelation preservation, and physical-bound violations
   between the imputed-only cells and the observed-only cells. Used to score
   the *current* ``emis_imputed.parquet`` and produce a baseline report.

2. ``artificial_mask_audit`` - requires a callable imputer
   ``Callable[[pd.DataFrame], pd.DataFrame]``. MCAR-masks observed cells in a
   chosen evaluation period, calls the imputer, compares imputed values at
   masked positions against ground truth. Reports per-column MAE / RMSE /
   sMAPE / Pearson r. Used to evaluate retrained models.

Both modes share the same ``ColumnReport`` dataclass and ``summarize_to_markdown``
formatter so reports are directly comparable across runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

# ---- physical-bound rules -----------------------------------------------------

GENERATION_NONNEG_COLS: tuple[str, ...] = (
    "gen_biomass", "gen_fossil_lignite", "gen_fossil_coal_gas", "gen_fossil_gas",
    "gen_fossil_hard_coal", "gen_fossil_oil", "gen_geothermal", "gen_hydro_ror",
    "gen_hydro_reservoir", "gen_nuclear", "gen_other_renewable", "gen_solar",
    "gen_waste", "gen_wind_offshore", "gen_wind_onshore", "gen_other",
)
# gen_hydro_pumped is signed (consumption negative, generation positive).

CROSS_BORDER_PAIRS: tuple[tuple[str, str], ...] = (
    ("FR_to_DELU", "DELU_to_FR"),
    ("AT_to_DELU", "DELU_to_AT"),
    ("CH_to_DELU", "DELU_to_CH"),
)


# ---- result schema ------------------------------------------------------------

@dataclass
class StaticColumnReport:
    """Per-column static-audit findings (no ground truth required)."""

    column: str
    nan_rate_raw: float
    n_observed: int
    n_imputed: int
    mean_observed: float | None
    mean_imputed: float | None
    std_observed: float | None
    std_imputed: float | None
    ks_pvalue: float | None
    wasserstein: float | None
    acf_observed: dict[int, float]
    acf_imputed: dict[int, float]
    physical_violations: dict[str, int]
    status: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class MaskedColumnReport:
    """Per-column artificial-mask audit findings (ground truth available)."""

    column: str
    n_evaluated: int
    mae: float
    rmse: float
    smape: float | None
    pearson: float | None


# ---- distribution / autocorrelation helpers -----------------------------------

def _ks_and_wasserstein(obs: np.ndarray, imp: np.ndarray) -> tuple[float | None, float | None]:
    """Two-sample KS p-value + 1-Wasserstein distance. Returns (None, None) if insufficient samples."""
    if len(obs) < 30 or len(imp) < 30:
        return None, None
    try:
        ks = float(stats.ks_2samp(obs, imp).pvalue)
        wd = float(stats.wasserstein_distance(obs, imp))
        return ks, wd
    except Exception:
        return None, None


def _safe_acf(values: np.ndarray, lags: Sequence[int]) -> dict[int, float]:
    """Autocorrelation at the requested lags. NaNs in ``values`` are handled by interpolation;
    if too few non-NaN values remain, returns NaN per lag."""
    finite = ~np.isnan(values)
    if finite.sum() < max(lags) * 4:
        return {lag: float("nan") for lag in lags}
    series = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    try:
        acvals = acf(series, nlags=max(lags), fft=True, missing="drop")
        return {lag: float(acvals[lag]) for lag in lags}
    except Exception:
        return {lag: float("nan") for lag in lags}


# ---- physical bound checks ----------------------------------------------------

def count_physical_violations(
    df: pd.DataFrame,
    column: str,
    sun_elevation: pd.Series | None = None,
) -> dict[str, int]:
    """Count physical-bound violations for a given column.

    Returns a dict ``{rule_name: violation_count}``. Empty dict if no rule applies.

    ``sun_elevation`` is the per-row sun elevation in degrees (positive when
    the sun is above the horizon). Required only for ``gen_solar``.
    """
    out: dict[str, int] = {}
    s = df[column]
    if column in GENERATION_NONNEG_COLS:
        out["negative_generation"] = int((s < -1e-3).sum())
    if column == "gen_solar" and sun_elevation is not None:
        # full night = sun more than 6° below horizon (civil twilight)
        night_mask = sun_elevation < -6.0
        out["solar_at_full_night"] = int(((s.abs() > 1.0) & night_mask).sum())
    if column in {pair[0] for pair in CROSS_BORDER_PAIRS} | {pair[1] for pair in CROSS_BORDER_PAIRS}:
        for a, b in CROSS_BORDER_PAIRS:
            if column == a and b in df.columns:
                out["both_directions_positive"] = int(((df[a] > 1.0) & (df[b] > 1.0)).sum())
                break
    return out


def compute_sun_elevation(
    index: pd.DatetimeIndex,
    lat: float = 51.0,
    lon: float = 10.0,
) -> pd.Series:
    """Approximate solar elevation (degrees) at the centre of Germany for each timestamp.

    Uses the NOAA solar position formula directly - much faster than calling
    astral per-row for ~60k timestamps.
    """
    if index.tz is None:
        utc_index = index.tz_localize("UTC")
    else:
        utc_index = index.tz_convert("UTC")
    # Day of year fraction
    day = utc_index.dayofyear + (utc_index.hour + utc_index.minute / 60) / 24
    gamma = 2 * np.pi / 365 * (day - 1)
    # Solar declination (radians, Spencer 1971)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    # Solar time (hours)
    time_offset = eqtime + 4 * lon
    tst = utc_index.hour * 60 + utc_index.minute + time_offset
    hour_angle = np.deg2rad(tst / 4 - 180)
    lat_rad = np.deg2rad(lat)
    elevation = np.arcsin(
        np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    )
    return pd.Series(np.rad2deg(elevation), index=index, name="sun_elevation")


# ---- static audit -------------------------------------------------------------

def _classify(report: StaticColumnReport) -> tuple[str, list[str]]:
    """Decide green/yellow/red status from the metrics."""
    reasons: list[str] = []
    status = "green"

    physical_total = sum(report.physical_violations.values())
    if physical_total > 100:
        status = "red"
        reasons.append(f"{physical_total} physical-bound violations")
    elif physical_total > 0:
        status = max(status, "yellow", key=("green", "yellow", "red").index)
        reasons.append(f"{physical_total} physical-bound violations")

    # Mean shift relative to observed std
    if (
        report.mean_observed is not None
        and report.mean_imputed is not None
        and report.std_observed is not None
        and report.std_observed > 0
    ):
        mean_shift = abs(report.mean_imputed - report.mean_observed) / report.std_observed
        if mean_shift > 1.0:
            status = "red"
            reasons.append(f"mean shift {mean_shift:.2f}σ vs observed")
        elif mean_shift > 0.5:
            status = max(status, "yellow", key=("green", "yellow", "red").index)
            reasons.append(f"mean shift {mean_shift:.2f}σ vs observed")

    # Sign flip: opposite signs of means
    if (
        report.mean_observed is not None
        and report.mean_imputed is not None
        and abs(report.mean_observed) > 1e-3
        and abs(report.mean_imputed) > 1e-3
        and np.sign(report.mean_observed) != np.sign(report.mean_imputed)
    ):
        status = "red"
        reasons.append("imputed mean has opposite sign vs observed")

    # Variance collapse
    if (
        report.std_observed is not None
        and report.std_imputed is not None
        and report.std_observed > 0
        and report.std_imputed / report.std_observed < 0.3
    ):
        status = "red"
        reasons.append(
            f"variance collapse ({report.std_imputed / report.std_observed:.2f}× of observed)"
        )

    # KS p-value below 0.01 means distributions differ
    if report.ks_pvalue is not None and report.ks_pvalue < 0.01:
        status = max(status, "yellow", key=("green", "yellow", "red").index)
        reasons.append(f"KS p={report.ks_pvalue:.2e} (distributions differ)")

    return status, reasons


def static_audit(
    imputed_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    acf_lags: Sequence[int] = (1, 24, 168),
) -> list[StaticColumnReport]:
    """Audit an imputed DataFrame against the raw NaN-bearing source.

    For each column with at least one NaN in ``raw_df`` (and present in both
    frames), compares the distribution of observed cells vs imputed cells, the
    autocorrelation at the given lags, and counts physical-bound violations.
    Columns fully observed in ``raw_df`` are skipped (nothing to audit).
    """
    if columns is None:
        columns = [
            c for c in imputed_df.columns
            if c in raw_df.columns and raw_df[c].isna().any()
        ]
    sun_elev = compute_sun_elevation(imputed_df.index)
    reports: list[StaticColumnReport] = []
    for col in columns:
        raw_mask = raw_df[col].isna()
        observed = imputed_df.loc[~raw_mask, col].to_numpy()
        imputed = imputed_df.loc[raw_mask, col].to_numpy()
        observed_clean = observed[~np.isnan(observed)]
        imputed_clean = imputed[~np.isnan(imputed)]

        ks_p, wd = _ks_and_wasserstein(observed_clean, imputed_clean)
        report = StaticColumnReport(
            column=col,
            nan_rate_raw=float(raw_mask.mean()),
            n_observed=int(len(observed_clean)),
            n_imputed=int(len(imputed_clean)),
            mean_observed=float(observed_clean.mean()) if len(observed_clean) else None,
            mean_imputed=float(imputed_clean.mean()) if len(imputed_clean) else None,
            std_observed=float(observed_clean.std()) if len(observed_clean) else None,
            std_imputed=float(imputed_clean.std()) if len(imputed_clean) else None,
            ks_pvalue=ks_p,
            wasserstein=wd,
            acf_observed=_safe_acf(np.where(raw_mask, np.nan, imputed_df[col].to_numpy()), acf_lags),
            acf_imputed=_safe_acf(imputed_df[col].to_numpy(), acf_lags),
            physical_violations=count_physical_violations(imputed_df, col, sun_elev),
            status="green",
        )
        report.status, report.reasons = _classify(report)
        reports.append(report)
    return reports


# ---- artificial-mask audit ----------------------------------------------------

def artificial_mask_audit(
    raw_df: pd.DataFrame,
    imputer: Callable[[pd.DataFrame], pd.DataFrame],
    eval_index: pd.DatetimeIndex,
    columns: Iterable[str] | None = None,
    mask_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[list[MaskedColumnReport], pd.DataFrame]:
    """Mask observed cells in the eval period, run the imputer, score per-column.

    Parameters
    ----------
    raw_df : full raw dataframe (NaN-bearing); the eval period must be a subset.
    imputer : callable taking a NaN-bearing DataFrame and returning an imputed one.
        Must preserve the index and columns. (Wrap your model with the convention
        you choose; this function does not assume any model class.)
    eval_index : the timestamps over which to introduce extra masks.
    columns : columns to evaluate. Defaults to every column with both observed
        cells in eval_index AND any NaN in raw_df.
    mask_ratio : fraction of observed cells to MCAR-mask per column.
    seed : RNG seed.

    Returns ``(reports, masked_df)`` so you can inspect what was actually masked.
    """
    rng = np.random.default_rng(seed)
    df = raw_df.copy()
    if columns is None:
        columns = [c for c in df.columns if df[c].isna().any()]

    mask_record = pd.DataFrame(False, index=df.index, columns=list(columns))
    for col in columns:
        observed_in_eval = df.loc[eval_index, col].notna()
        observed_idx = observed_in_eval[observed_in_eval].index
        n_to_mask = int(len(observed_idx) * mask_ratio)
        if n_to_mask == 0:
            continue
        chosen = rng.choice(observed_idx, size=n_to_mask, replace=False)
        mask_record.loc[chosen, col] = True
        df.loc[chosen, col] = np.nan

    imputed = imputer(df)
    if not imputed.index.equals(df.index):
        raise ValueError("imputer must preserve the index")

    reports: list[MaskedColumnReport] = []
    for col in columns:
        m = mask_record[col]
        if not m.any():
            continue
        truth = raw_df.loc[m, col].to_numpy()
        pred = imputed.loc[m, col].to_numpy()
        valid = ~(np.isnan(truth) | np.isnan(pred))
        truth, pred = truth[valid], pred[valid]
        if len(truth) == 0:
            continue
        mae = float(np.abs(truth - pred).mean())
        rmse = float(np.sqrt(((truth - pred) ** 2).mean()))
        denom = (np.abs(truth) + np.abs(pred)) / 2
        smape = float((np.abs(truth - pred) / np.where(denom > 1e-9, denom, 1)).mean()) if denom.mean() > 0 else None
        pearson = float(np.corrcoef(truth, pred)[0, 1]) if truth.std() > 0 and pred.std() > 0 else None
        reports.append(MaskedColumnReport(
            column=col,
            n_evaluated=int(len(truth)),
            mae=mae,
            rmse=rmse,
            smape=smape,
            pearson=pearson,
        ))
    return reports, mask_record


# ---- block-mask audit ---------------------------------------------------------

def block_mask_audit(
    raw_df: pd.DataFrame,
    imputer: Callable[[pd.DataFrame], pd.DataFrame],
    eval_index: pd.DatetimeIndex,
    columns: Iterable[str] | None = None,
    block_hours: int = 24,
    n_blocks: int = 30,
    seed: int = 0,
) -> tuple[list[MaskedColumnReport], pd.DataFrame]:
    """Audit imputer accuracy when masks form *contiguous blocks*, not MCAR.

    Real missingness in this dataset is bursty (exchange holidays for
    ``carbon_ets``, full-day gaps in some generation streams). MCAR masking
    underestimates the real difficulty of imputation because it never asks the
    model to interpolate through a 24h+ run. This complement places ``n_blocks``
    contiguous blocks of length ``block_hours`` per column, computes the same
    MAE/RMSE/sMAPE/Pearson stats, and returns the block mask record so the
    caller can inspect exactly what was masked.

    Reference: time-series-analysis skill's emphasis on multi-step ahead
    forecasts - block masking is the closest equivalent for imputation.
    """
    rng = np.random.default_rng(seed)
    df = raw_df.copy()
    if columns is None:
        columns = [c for c in df.columns if df[c].isna().any()]

    mask_record = pd.DataFrame(False, index=df.index, columns=list(columns))
    eval_index = pd.DatetimeIndex(eval_index)
    eval_positions = df.index.get_indexer(eval_index)
    eval_positions = eval_positions[eval_positions >= 0]
    max_start = eval_positions.max() - block_hours + 1
    if max_start <= eval_positions.min():
        raise ValueError(
            f"eval_index too short for block_hours={block_hours} "
            f"(need ≥ {block_hours + 1} contiguous hours)"
        )

    for col in columns:
        col_idx = df.columns.get_loc(col)
        starts = rng.integers(eval_positions.min(), max_start, size=n_blocks)
        placed_blocks = 0
        for start in starts:
            block_slice = slice(start, start + block_hours)
            # Only mask cells that were originally observed (skip pre-existing NaN
            # so we have ground truth).
            originally_observed = df.iloc[block_slice, col_idx].notna()
            if originally_observed.sum() == 0:
                continue
            for offset in np.where(originally_observed.to_numpy())[0]:
                pos = start + offset
                mask_record.iat[pos, mask_record.columns.get_loc(col)] = True
                df.iat[pos, col_idx] = np.nan
            placed_blocks += 1
        if placed_blocks == 0:
            # Could not place any block (column entirely missing in eval window).
            mask_record.drop(columns=col, inplace=True)

    imputed = imputer(df)
    if not imputed.index.equals(df.index):
        raise ValueError("imputer must preserve the index")

    reports: list[MaskedColumnReport] = []
    for col in mask_record.columns:
        m = mask_record[col]
        if not m.any():
            continue
        truth = raw_df.loc[m, col].to_numpy()
        pred = imputed.loc[m, col].to_numpy()
        valid = ~(np.isnan(truth) | np.isnan(pred))
        truth, pred = truth[valid], pred[valid]
        if len(truth) == 0:
            continue
        mae = float(np.abs(truth - pred).mean())
        rmse = float(np.sqrt(((truth - pred) ** 2).mean()))
        denom = (np.abs(truth) + np.abs(pred)) / 2
        smape = float((np.abs(truth - pred) / np.where(denom > 1e-9, denom, 1)).mean()) if denom.mean() > 0 else None
        pearson = float(np.corrcoef(truth, pred)[0, 1]) if truth.std() > 0 and pred.std() > 0 else None
        reports.append(MaskedColumnReport(
            column=col, n_evaluated=int(len(truth)),
            mae=mae, rmse=rmse, smape=smape, pearson=pearson,
        ))
    return reports, mask_record


# ---- markdown formatting ------------------------------------------------------

_STATUS_GLYPH = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


def summarize_static_to_markdown(
    reports: list[StaticColumnReport],
    title: str = "Imputation Static Audit",
    notes: str | None = None,
) -> str:
    rows = sorted(reports, key=lambda r: -r.nan_rate_raw)
    lines = [
        f"# {title}",
        "",
        f"_Generated against {len(rows)} columns with NaN in the raw source._",
        "",
    ]
    if notes:
        lines.extend([notes, ""])
    lines += [
        "| Status | Column | NaN rate | mean(obs) | mean(imp) | std(obs) | std(imp) | KS p | Wasserstein | Physical issues | Reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        physical = "; ".join(f"{k}={v}" for k, v in r.physical_violations.items() if v)
        reasons = "; ".join(r.reasons)
        lines.append(
            f"| {_STATUS_GLYPH[r.status]} | `{r.column}` | {r.nan_rate_raw:.1%} | "
            f"{r.mean_observed if r.mean_observed is None else f'{r.mean_observed:.2f}'} | "
            f"{r.mean_imputed if r.mean_imputed is None else f'{r.mean_imputed:.2f}'} | "
            f"{r.std_observed if r.std_observed is None else f'{r.std_observed:.2f}'} | "
            f"{r.std_imputed if r.std_imputed is None else f'{r.std_imputed:.2f}'} | "
            f"{r.ks_pvalue if r.ks_pvalue is None else f'{r.ks_pvalue:.2e}'} | "
            f"{r.wasserstein if r.wasserstein is None else f'{r.wasserstein:.2f}'} | "
            f"{physical or '-'} | {reasons or '-'} |"
        )
    counts = {"green": 0, "yellow": 0, "red": 0}
    for r in rows:
        counts[r.status] += 1
    lines += [
        "",
        f"**Summary:** {_STATUS_GLYPH['green']} {counts['green']} | "
        f"{_STATUS_GLYPH['yellow']} {counts['yellow']} | "
        f"{_STATUS_GLYPH['red']} {counts['red']}",
    ]
    return "\n".join(lines)


def summarize_masked_to_markdown(
    reports: list[MaskedColumnReport],
    title: str = "Imputation Artificial-Mask Audit",
) -> str:
    rows = sorted(reports, key=lambda r: -r.mae)
    lines = [
        f"# {title}",
        "",
        f"_Per-column accuracy on MCAR-masked observed cells._",
        "",
        "| Column | n | MAE | RMSE | sMAPE | Pearson |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.column}` | {r.n_evaluated} | {r.mae:.4f} | {r.rmse:.4f} | "
            f"{'-' if r.smape is None else f'{r.smape:.3f}'} | "
            f"{'-' if r.pearson is None else f'{r.pearson:.3f}'} |"
        )
    return "\n".join(lines)


def reports_to_json(reports: list[StaticColumnReport] | list[MaskedColumnReport]) -> str:
    return json.dumps([asdict(r) for r in reports], indent=2, default=float)


# ---- CLI ----------------------------------------------------------------------

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Audit an imputed parquet against the raw source.")
    parser.add_argument("--raw", type=Path, default=Path("data/processed/emis_raw.parquet"))
    parser.add_argument("--imputed", type=Path, default=Path("data/processed/emis_imputed.parquet"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/imputation_audit_baseline.md"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/imputation_audit_baseline.json"))
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw)
    imp = pd.read_parquet(args.imputed)
    reports = static_audit(imp, raw)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(summarize_static_to_markdown(reports))
    args.out_json.write_text(reports_to_json(reports))
    print(f"Wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    _cli()
