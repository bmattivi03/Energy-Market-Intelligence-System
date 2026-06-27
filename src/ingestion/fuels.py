import time
import requests
import pandas as pd
import yfinance as yf

# Energy-Charts.info (Fraunhofer ISE) — free, no API key, CC BY 4.0
_EC_BASE = "https://api.energy-charts.info"
_EC_CO2_ENDPOINT = "/co2_price"          # returns daily EUA settlement prices
_EC_CO2_PARAMS   = {"bzn": "DE-LU"}      # bidding zone; try country=de as fallback

_EC_RETRY_DELAYS = [10, 30, 60]          # seconds between retries on 429/503


def _fetch_energy_charts_co2(start_date: str, end_date: str) -> pd.Series:
    """
    Fetch daily EUA/CO2 settlement prices from Energy-Charts.info.
    Returns a Series indexed by date (UTC, tz-aware). Returns empty Series on failure.
    Falls back to country=de if bzn=DE-LU returns 404.
    """
    param_variants = [
        {**_EC_CO2_PARAMS, "start": start_date, "end": end_date},
        {"country": "de", "start": start_date, "end": end_date},
    ]

    for params in param_variants:
        for attempt, delay in enumerate([0] + _EC_RETRY_DELAYS):
            if delay:
                print(f"    [energy-charts] rate limited, retrying in {delay}s …")
                time.sleep(delay)
            try:
                r = requests.get(
                    _EC_BASE + _EC_CO2_ENDPOINT,
                    params=params,
                    timeout=20,
                    headers={"User-Agent": "EMIS-ingestion/1.0"},
                )
                if r.status_code == 404:
                    break                    # try next param variant
                if r.status_code == 429 and attempt < len(_EC_RETRY_DELAYS):
                    continue                 # retry after delay
                r.raise_for_status()

                data = r.json()
                # Expected shape: {"unix_seconds": [...], "price": [...], "unit": "EUR/t CO2"}
                if "unix_seconds" not in data or "price" not in data:
                    print(f"    [energy-charts] unexpected response keys: {list(data.keys())}")
                    break

                idx = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)
                s = pd.Series(data["price"], index=idx, name="carbon_ets", dtype=float)
                s = s.resample("1D").last()          # one value per day
                print(f"    [energy-charts] fetched {len(s)} days  "
                      f"{s.index[0].date()} -> {s.index[-1].date()}")
                return s

            except requests.RequestException as e:
                print(f"    [energy-charts] attempt {attempt + 1} failed: {e}")

    print("    [energy-charts] all attempts failed — carbon ETS will use KEUA fallback only")
    return pd.Series(dtype=float, name="carbon_ets")


def fetch_fuel_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily fuel prices for the EMIS pipeline.

    Sources
    -------
    ttf_gas   : Yahoo Finance TTF=F  (Dutch TTF natural gas front-month)
    carbon_ets: Energy-Charts.info   (EUA daily settlement, Fraunhofer ISE)
                + Yahoo Finance KEUA (KraneShares EU Carbon ETF, listed 2021-09-29)
                  used as supplement / fallback for post-Sep-2021 period

    Carbon coverage notes
    ---------------------
    - Energy-Charts covers 2010+ (no API key, CC BY 4.0).
    - KEUA starts 2021-09-29; used to fill any gap Energy-Charts leaves.
    - Remaining NaN (if any) is left for Glocal-IB imputation — the 2019-2021
      period is MAR (missing because the ETF didn't exist, not price-driven)
      and correlates strongly with electricity price and TTF gas.
    """
    frames = {}

    def _to_utc_series(s: pd.Series) -> pd.Series:
        idx = s.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        s = s.copy()
        s.index = idx
        return s

    # ── TTF natural gas ────────────────────────────────────────────────────────
    print(f"Fetching ttf_gas (TTF=F) from {start_date} to {end_date}...")
    ttf = yf.download("TTF=F", start=start_date, end=end_date, interval="1d",
                      progress=False, auto_adjust=True)
    if not ttf.empty:
        frames["ttf_gas"] = _to_utc_series(ttf["Close"].squeeze().rename("ttf_gas"))
    else:
        print("  [WARN] TTF=F returned no data")

    # ── Carbon ETS — primary: Energy-Charts.info ───────────────────────────────
    print(f"Fetching carbon_ets (Energy-Charts CO2 price) from {start_date} to {end_date}...")
    ec_co2 = _fetch_energy_charts_co2(start_date, end_date)

    # ── Carbon ETS — supplement: KEUA (2021-09-29 onward) ─────────────────────
    print(f"Fetching carbon_ets supplement (KEUA) from {start_date} to {end_date}...")
    keua = yf.download("KEUA", start=start_date, end=end_date, interval="1d",
                       progress=False, auto_adjust=True)
    if not keua.empty:
        keua_s = _to_utc_series(keua["Close"].squeeze().rename("carbon_ets"))
    else:
        keua_s = pd.Series(dtype=float, name="carbon_ets")

    # Merge: Energy-Charts as base, fill gaps with KEUA (normalised to EC scale)
    if not ec_co2.empty and not keua_s.empty:
        # Align on daily UTC index
        overlap = ec_co2.index.intersection(keua_s.index)
        if len(overlap) > 30:
            # Scale KEUA (USD ETF NAV) to EC price level via median ratio
            ratio = ec_co2.reindex(overlap).median() / keua_s.reindex(overlap).median()
            keua_scaled = keua_s * ratio
        else:
            keua_scaled = keua_s

        carbon = ec_co2.copy()
        gap_mask = carbon.isna() | ~carbon.index.isin(ec_co2.dropna().index)
        carbon = carbon.combine_first(keua_scaled)
        frames["carbon_ets"] = carbon

    elif not ec_co2.empty:
        frames["carbon_ets"] = ec_co2

    elif not keua_s.empty:
        print("  [WARN] Energy-Charts unavailable — using KEUA only (coverage from 2021-09-29)")
        frames["carbon_ets"] = keua_s

    else:
        print("  [WARN] No carbon ETS data retrieved")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames.values(), axis=1)
    result.columns = list(frames.keys())
    result = result.sort_index()

    # Coverage report
    for col in result.columns:
        nan_pct = result[col].isna().mean()
        first = result[col].first_valid_index()
        last  = result[col].last_valid_index()
        status = "[OK]" if nan_pct < 0.05 else "[WARN]"
        print(f"  {status} {col}: {len(result[col].dropna())} days  "
              f"{first.date() if first else 'N/A'} -> {last.date() if last else 'N/A'}  "
              f"NaN={nan_pct:.1%}")

    return result


if __name__ == "__main__":
    fuels = fetch_fuel_prices("2019-01-01", "2025-12-31")
    if not fuels.empty:
        out = "data/raw/fuels_2019_2025.parquet"
        fuels.to_parquet(out)
        print(f"Saved {out}  ({len(fuels):,} rows)")
