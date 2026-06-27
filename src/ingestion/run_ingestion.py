import os
import sys
import json
import argparse
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from src.ingestion.entsoe import fetch_range, fetch_crossborder_range, DE_LU, DE_LU_MARKET
from src.ingestion.weather import fetch_weather
from src.ingestion.fuels import fetch_fuel_prices

START_YEAR = 2019
END_YEAR   = 2025
START_DATE = "2019-01-01"
END_DATE   = "2025-12-31"
RAW_DIR    = "data/raw"

CHECKPOINT_FILE = os.path.join(RAW_DIR, ".checkpoint.json")

PSR_NAMES = {
    "B01": "gen_biomass",
    "B02": "gen_fossil_lignite",
    "B03": "gen_fossil_coal_gas",
    "B04": "gen_fossil_gas",
    "B05": "gen_fossil_hard_coal",
    "B06": "gen_fossil_oil",
    "B09": "gen_geothermal",
    "B10": "gen_hydro_pumped",
    "B11": "gen_hydro_ror",
    "B12": "gen_hydro_reservoir",
    "B14": "gen_nuclear",
    "B15": "gen_other_renewable",
    "B16": "gen_solar",
    "B17": "gen_waste",
    "B18": "gen_wind_offshore",
    "B19": "gen_wind_onshore",
    "B20": "gen_other",
}

CITIES = {
    "Berlin":    (52.52, 13.41),
    "Hamburg":   (53.55,  9.99),
    "Munich":    (48.13, 11.58),
    "Cologne":   (50.93,  6.95),
    "Frankfurt": (50.11,  8.68),
}


def ensure_dirs():
    for d in [RAW_DIR, "data/processed", "data/splits"]:
        os.makedirs(d, exist_ok=True)


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def mark_done(checkpoint: dict, step: str):
    checkpoint[step] = True
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def validate_output(df: pd.DataFrame, name: str) -> bool:
    ok = True
    rows = len(df)

    # Resolve min/max date — handles both flat DatetimeIndex and MultiIndex (e.g. weather)
    min_date = max_date = None
    if isinstance(df.index, pd.MultiIndex):
        for level_idx in range(df.index.nlevels):
            vals = df.index.get_level_values(level_idx)
            if pd.api.types.is_datetime64_any_dtype(vals):
                min_date = vals.min()
                max_date = vals.max()
                break
    else:
        min_date = df.index.min()
        max_date = df.index.max()

    # fuels is daily (~1,500 rows for 6 years); everything else is hourly/sub-hourly
    min_rows = 1_400 if name == "fuels" else 50_000
    if rows < min_rows:
        print(f"  [WARN] {name}: only {rows:,} rows (expected ~{min_rows:,}+)")
        ok = False

    if min_date is not None:
        expected_start = pd.Timestamp("2019-01-01", tz="UTC")
        expected_end   = pd.Timestamp("2025-12-30", tz="UTC")
        # Normalize tz for comparison
        if min_date.tzinfo is None:
            expected_start = expected_start.tz_localize(None)
            expected_end   = expected_end.tz_localize(None)
        if min_date > expected_start + pd.Timedelta(days=2):
            print(f"  [WARN] {name}: coverage starts at {min_date.date()} (expected ~2019-01-01)")
            ok = False
        if max_date < expected_end:
            print(f"  [WARN] {name}: coverage ends at {max_date.date()} (expected ~2025-12-31)")
            ok = False

    nan_pct = df.isna().mean()
    bad_cols = nan_pct[nan_pct > 0.5]
    if not bad_cols.empty:
        print(f"  [WARN] {name}: columns >50% NaN: {bad_cols.to_dict()}")
        ok = False

    date_str = f"{min_date.date()} → {max_date.date()}" if min_date is not None else "no date index"
    if ok:
        print(f"  [OK] {name}: {rows:,} rows, {date_str}, max NaN {nan_pct.max():.1%}")
    return ok


def print_final_report():
    print("\n" + "=" * 70)
    print(f"{'File':<45} {'Rows':>8}  {'From':<12} {'To':<12} {'MaxNaN':>7}")
    print("=" * 70)
    for path in sorted(glob.glob(f"{RAW_DIR}/*.parquet")):
        try:
            df = pd.read_parquet(path)
            name = os.path.basename(path)
            rows = len(df)
            if isinstance(df.index, pd.MultiIndex):
                min_d = max_d = None
                for lvl in range(df.index.nlevels):
                    vals = df.index.get_level_values(lvl)
                    if pd.api.types.is_datetime64_any_dtype(vals):
                        min_d, max_d = vals.min(), vals.max()
                        break
            else:
                min_d, max_d = df.index.min(), df.index.max()
            max_nan = df.isna().mean().max()
            date_str = f"{str(min_d.date()):<12} {str(max_d.date()):<12}" if min_d is not None else f"{'N/A':<12} {'N/A':<12}"
            print(f"{name:<45} {rows:>8,}  {date_str} {max_nan:>6.1%}")
        except Exception as e:
            print(f"{os.path.basename(path):<45} ERROR: {e}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run EMIS data ingestion")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint and re-fetch everything")
    args = parser.parse_args()

    ensure_dirs()

    if args.fresh and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("[INFO] Checkpoint cleared — re-fetching all sources.")

    checkpoint = load_checkpoint()

    # --- ENTSO-E Day-Ahead Prices ---
    print("\n=== [1/6] ENTSO-E Day-Ahead Prices ===")
    if checkpoint.get("prices"):
        print("  [SKIP] Already complete.")
    else:
        prices = fetch_range(START_YEAR, END_YEAR, "A44", {
            "in_Domain": DE_LU_MARKET, "out_Domain": DE_LU_MARKET,
        })
        if not prices.empty:
            out = f"{RAW_DIR}/entsoe_prices_2019_2025.parquet"
            prices.to_parquet(out)
            print(f"Saved {out}  ({len(prices):,} rows)")
            validate_output(prices, "entsoe_prices")
            mark_done(checkpoint, "prices")

    # --- ENTSO-E Actual Load ---
    print("\n=== [2/6] ENTSO-E Actual Load ===")
    if checkpoint.get("load"):
        print("  [SKIP] Already complete.")
    else:
        load = fetch_range(START_YEAR, END_YEAR, "A65", {
            "processType": "A16", "outBiddingZone_Domain": DE_LU,
        })
        if not load.empty:
            out = f"{RAW_DIR}/entsoe_load_2019_2025.parquet"
            load.to_parquet(out)
            print(f"Saved {out}  ({len(load):,} rows)")
            validate_output(load, "entsoe_load")
            mark_done(checkpoint, "load")

    # --- ENTSO-E Generation per Source ---
    print("\n=== [3/6] ENTSO-E Generation per Source ===")
    if checkpoint.get("generation"):
        print("  [SKIP] Already complete.")
    else:
        all_gen_dfs = []
        for psr_code, psr_name in PSR_NAMES.items():
            print(f"  Fetching {psr_name} ({psr_code})...")
            df = fetch_range(START_YEAR, END_YEAR, "A75", {
                "processType": "A16",
                "in_Domain": DE_LU,
                "out_Domain": DE_LU,
                "psrType": psr_code
            }, chunk_days=90)

            if not df.empty:
                df = df[["value"]].rename(columns={"value": psr_name})
                df = df[~df.index.duplicated(keep="last")]
                all_gen_dfs.append(df)

        if all_gen_dfs:
            gen_combined = pd.concat(all_gen_dfs, axis=1).sort_index()
            out = f"{RAW_DIR}/entsoe_generation_2019_2025.parquet"
            gen_combined.to_parquet(out)
            print(f"Saved {out} ({len(gen_combined):,} rows, {len(gen_combined.columns)} sources)")
            validate_output(gen_combined, "entsoe_generation")
            mark_done(checkpoint, "generation")
        else:
            print("Warning: No generation data fetched.")

    # --- ENTSO-E Cross-Border Flows ---
    print("\n=== [4/6] ENTSO-E Cross-Border Flows ===")
    if checkpoint.get("crossborder"):
        print("  [SKIP] Already complete.")
    else:
        cb_raw = fetch_crossborder_range(START_YEAR, END_YEAR)
        if not cb_raw.empty:
            # Pivot to wide format — each direction becomes a column.
            # Different borders report at different resolutions (AT/CH at 15-min,
            # FR at 60-min with gaps). Resample everything to 1H so the index is
            # consistent and FR columns don't end up 84% NaN from the mismatch.
            cb_pivot = (
                cb_raw.reset_index()
                .pivot_table(index="datetime_utc", columns="direction", values="value", aggfunc="sum")
            )
            cb_pivot.columns.name = None
            cb_pivot = cb_pivot.resample("1h").sum(min_count=1)  # NaN if truly no data in the hour
            for neighbor in ["FR", "AT", "CH"]:
                into    = f"{neighbor}_to_DELU"
                out_col = f"DELU_to_{neighbor}"
                if into in cb_pivot.columns and out_col in cb_pivot.columns:
                    cb_pivot[f"net_import_{neighbor}"] = cb_pivot[into] - cb_pivot[out_col]
            out = f"{RAW_DIR}/entsoe_crossborder_2019_2025.parquet"
            cb_pivot.to_parquet(out)
            print(f"Saved {out}  ({len(cb_pivot):,} rows)")
            validate_output(cb_pivot, "entsoe_crossborder")
            mark_done(checkpoint, "crossborder")

    # --- Open-Meteo Weather ---
    print("\n=== [5/6] Open-Meteo Weather ===")
    if checkpoint.get("weather"):
        print("  [SKIP] Already complete.")
    else:
        weather = fetch_weather(START_DATE, END_DATE, CITIES)
        if not weather.empty:
            out = f"{RAW_DIR}/weather_2019_2025.parquet"
            weather.to_parquet(out)
            print(f"Saved {out}  ({len(weather):,} rows across {len(CITIES)} cities)")
            validate_output(weather, "weather")
            mark_done(checkpoint, "weather")

    # --- yfinance Fuel Prices ---
    print("\n=== [6/6] yfinance Fuel Prices ===")
    if checkpoint.get("fuels"):
        print("  [SKIP] Already complete.")
    else:
        fuels = fetch_fuel_prices(START_DATE, END_DATE)
        if not fuels.empty:
            out = f"{RAW_DIR}/fuels_2019_2025.parquet"
            fuels.to_parquet(out)
            print(f"Saved {out}  ({len(fuels):,} rows)")
            ok = validate_output(fuels, "fuels")
            # Only mark done if carbon_ets has <50% NaN — otherwise keep retryable
            carbon_nan = fuels["carbon_ets"].isna().mean() if "carbon_ets" in fuels.columns else 1.0
            if ok and carbon_nan < 0.50:
                mark_done(checkpoint, "fuels")
            else:
                print("  [INFO] fuels not checkpointed — carbon_ets NaN too high, will retry next run")

    print("\n=== Ingestion complete ===")
    print_final_report()


if __name__ == "__main__":
    main()
