import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
from typing import Optional

load_dotenv()

BASE_URL = "https://web-api.tp.entsoe.eu/api"
TOKEN = os.getenv("ENTSOE_TOKEN", "").strip()
DE_LU        = "10Y1001A1001A83F"  # physical / operational data (load, generation, flows)
DE_LU_MARKET = "10Y1001A1001A82H"  # market price zone (day-ahead prices A44)
DE_AT_LU     = "10Y1001A1001A63L"  # historical joint DE-AT-LU bidding zone (pre-Oct 2018/2019 data)

# Namespaces for ENTSO-E XMLs (can vary by document type)
NS = {
    "price": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3",
    "load": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0",
    "gen": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
}

_LOG_PATH = os.path.join("data", "raw", "ingestion.log")


def _log_failure(msg: str):
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    with open(_LOG_PATH, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()} {msg}\n")


def fetch_entsoe_data(params: dict, max_retries: int = 3) -> Optional[str]:
    params["securityToken"] = TOKEN
    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, params=params, timeout=120)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Network error ({exc.__class__.__name__}), retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            print(f"  Network error after {max_retries} attempts: {exc}")
            return None

        if response.status_code == 200:
            return response.text
        if response.status_code in (504, 503, 502) and attempt < max_retries - 1:
            wait = 30 * (attempt + 1)
            print(f"  Got {response.status_code}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            continue
        if "User is not authorized" in response.text:
            print(f"\n[CRITICAL] ENTSO-E Authorization Failed: The token is either invalid or your account hasn't been granted API access yet.")
            print("Please ensure you have requested API access by emailing transparency@entsoe.eu with 'API access' in the subject.")
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
        return None
    return None


def parse_entsoe_xml(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    ns_url = root.tag.split('}')[0].strip('{')
    ns = {"ns": ns_url}

    records = []
    for ts in root.findall(".//ns:TimeSeries", ns):
        psr_type = ts.find("ns:MktPSRType/ns:psrType", ns)
        psr_val = psr_type.text if psr_type is not None else None

        for period in ts.findall("ns:Period", ns):
            start_str = period.find("ns:timeInterval/ns:start", ns).text
            start_dt = pd.to_datetime(start_str).tz_convert("UTC")

            resolution = period.find("ns:resolution", ns).text
            res_minutes = 60 if "60" in resolution else 15

            for point in period.findall("ns:Point", ns):
                pos = int(point.find("ns:position", ns).text)
                val_node = point.find("ns:price.amount", ns)
                if val_node is None:
                    val_node = point.find("ns:quantity", ns)

                if val_node is not None:
                    qty = float(val_node.text)
                    dt = start_dt + timedelta(minutes=(pos - 1) * res_minutes)
                    record = {"datetime_utc": dt, "value": qty}
                    if psr_val:
                        record["psr_type"] = psr_val
                    records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df.drop_duplicates(subset=["datetime_utc", "psr_type"] if "psr_type" in df.columns else ["datetime_utc"])


def fetch_range(start_year: int, end_year: int, doc_type: str, extra_params: dict = None, chunk_days: int = 365) -> pd.DataFrame:
    all_dfs = []
    current_start = datetime(start_year, 1, 1)
    final_end = datetime(end_year, 12, 31, 23, 59)

    while current_start < final_end:
        current_end = min(current_start + timedelta(days=chunk_days), final_end)

        params = {
            "documentType": doc_type,
            "periodStart": current_start.strftime("%Y%m%d%H%M"),
            "periodEnd": current_end.strftime("%Y%m%d%H%M"),
        }
        if extra_params:
            params.update(extra_params)

        print(f"Fetching {doc_type} from {params['periodStart']} to {params['periodEnd']}...")
        xml_data = fetch_entsoe_data(params)
        if xml_data:
            df = parse_entsoe_xml(xml_data)
            if not df.empty:
                all_dfs.append(df)
            else:
                msg = f"WARN empty_response doc_type={doc_type} start={params['periodStart']} end={params['periodEnd']} extra={extra_params}"
                print(f"  [WARN] Empty response for chunk {params['periodStart']}-{params['periodEnd']}")
                _log_failure(msg)
        else:
            msg = f"WARN fetch_failed doc_type={doc_type} start={params['periodStart']} end={params['periodEnd']} extra={extra_params}"
            print(f"  [WARN] Fetch failed for chunk {params['periodStart']}-{params['periodEnd']}")
            _log_failure(msg)

        current_start = current_end + timedelta(minutes=1)
        time.sleep(1)  # Be nice to the API

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs).set_index("datetime_utc").sort_index()


BORDERS = {
    "FR": "10YFR-RTE------C",
    "AT": "10YAT-APG------L",
    "CH": "10YCH-SWISSGRIDZ",
}


def fetch_crossborder_range(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch physical cross-border flows between Germany (DE-LU) and FR, AT, CH.

    DE_AT_LU is confirmed dead for A11 (returns empty for all border pairs);
    only DE_LU EIC carries actual physical flow data.
    """
    all_flows = []

    for neighbor, neighbor_eic in BORDERS.items():
        for label, in_d, out_d in [
            (f"{neighbor}_to_DELU", neighbor_eic, DE_LU),
            (f"DELU_to_{neighbor}", DE_LU, neighbor_eic),
        ]:
            print(f"  Fetching {label}...")
            df = fetch_range(start_year, end_year, "A11", {
                "in_Domain": in_d,
                "out_Domain": out_d,
            }, chunk_days=90)
            if not df.empty:
                df = df[["value"]].copy()
                df["direction"] = label
                all_flows.append(df)

    if not all_flows:
        return pd.DataFrame()

    combined = (
        pd.concat(all_flows)
        .reset_index()
        .drop_duplicates(subset=["datetime_utc", "direction"])
        .set_index("datetime_utc")
        .sort_index()
    )
    return combined


if __name__ == "__main__":
    df = fetch_range(2024, 2024, "A44", {"in_Domain": DE_LU, "out_Domain": DE_LU})
    print(f"Test fetch: {len(df)} price records for 2024")
