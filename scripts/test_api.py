#!/usr/bin/env python3
"""
ENTSO-E API diagnostic script.
Makes real API calls with short date windows and shows exactly what comes back.
Run: python test_api.py
"""
import os
import sys
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://web-api.tp.entsoe.eu/api"
TOKEN    = os.getenv("ENTSOE_TOKEN", "").strip()

DE_LU        = "10Y1001A1001A83F"
DE_LU_MARKET = "10Y1001A1001A82H"
DE_AT_LU     = "10Y1001A1001A63L"
FR_EIC       = "10YFR-RTE------C"
AT_EIC       = "10YAT-APG------L"
CH_EIC       = "10YCH-SWISSGRIDZ"

RESULTS = []


def call(label: str, params: dict) -> dict:
    """Make one API call, return a summary dict."""
    p = dict(params)
    p["securityToken"] = TOKEN

    try:
        r = requests.get(BASE_URL, params=p, timeout=60)
    except Exception as exc:
        result = {"label": label, "status": "NET_ERROR", "http": None, "records": 0,
                  "note": str(exc)[:120]}
        RESULTS.append(result)
        return result

    http = r.status_code
    body = r.text

    # Count TimeSeries elements as a proxy for record richness
    ts_count = body.count("<TimeSeries>") + body.count("<TimeSeries ")
    # Count Point elements
    pt_count = body.count("<Point>") + body.count("<Point ")

    if http == 200 and pt_count == 0 and ts_count == 0:
        status = "EMPTY"
    elif http == 200:
        status = "OK"
    else:
        status = f"HTTP_{http}"

    # Show first 600 chars of body for debugging
    preview = body[:600].replace("\n", " ").replace("  ", " ")

    result = {
        "label": label,
        "status": status,
        "http": http,
        "time_series": ts_count,
        "points": pt_count,
        "preview": preview,
        "params_sent": {k: v for k, v in p.items() if k != "securityToken"},
    }
    RESULTS.append(result)
    return result


def section(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print('=' * 65)


def report(r: dict):
    icon = {"OK": "[OK  ]", "EMPTY": "[EMPT]"}.get(r["status"], "[FAIL]")
    ts   = r.get("time_series", "-")
    pts  = r.get("points", "-")
    print(f"\n{icon} {r['label']}")
    print(f"  status={r['status']}  http={r['http']}  TimeSeries={ts}  Points={pts}")
    if r["status"] != "OK":
        print(f"  preview: {r.get('preview', r.get('note', ''))[:300]}")


# ── Date windows ────────────────────────────────────────────────────────────
W_RECENT  = ("202301010000", "202302010000")   # Jan 2023 - after Ukraine war, before nuclear shutdown
W_OLD     = ("202001010000", "202002010000")   # Jan 2020 - pandemic year
W_NUCLEAR = ("202304010000", "202305010000")   # April 2023 - month of nuclear shutdown

if not TOKEN:
    print("[CRITICAL] ENTSOE_TOKEN not set. Export it before running.")
    sys.exit(1)

print(f"Token ends in: ...{TOKEN[-6:]}")
print(f"Base URL: {BASE_URL}")


# ──────────────────────────────────────────────────────────────────────────
section("BASELINE - A44 Prices & A65 Load (Jan 2023)")
# ──────────────────────────────────────────────────────────────────────────

call("A44 prices (DE_LU_MARKET)", {
    "documentType": "A44",
    "in_Domain": DE_LU_MARKET,
    "out_Domain": DE_LU_MARKET,
    "periodStart": W_RECENT[0],
    "periodEnd":   W_RECENT[1],
})

call("A65 load (DE_LU)", {
    "documentType": "A65",
    "processType": "A16",
    "outBiddingZone_Domain": DE_LU,
    "periodStart": W_RECENT[0],
    "periodEnd":   W_RECENT[1],
})


# ──────────────────────────────────────────────────────────────────────────
section("A75 GENERATION - Testing problem PSR types")
# ──────────────────────────────────────────────────────────────────────────

GEN_BASE = {
    "documentType": "A75",
    "processType":  "A16",
    "in_Domain":    DE_LU,
    "out_Domain":   DE_LU,
}

# B04 fossil gas - should always have data
call("A75 B04 fossil_gas (Jan 2023, should be full)", {**GEN_BASE, "psrType": "B04",
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})

# B03 fossil_coal_gas - was empty for ~2020-2022 in the big run
call("A75 B03 fossil_coal_gas (Jan 2020, was EMPTY)", {**GEN_BASE, "psrType": "B03",
     "periodStart": W_OLD[0], "periodEnd": W_OLD[1]})
call("A75 B03 fossil_coal_gas (Jan 2023, had data)", {**GEN_BASE, "psrType": "B03",
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})

# B14 nuclear - should be empty from April 2023
call("A75 B14 nuclear (Jan 2023, last operating months)", {**GEN_BASE, "psrType": "B14",
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})
call("A75 B14 nuclear (Apr 2023, shutdown month - expect EMPTY)", {**GEN_BASE, "psrType": "B14",
     "periodStart": W_NUCLEAR[0], "periodEnd": W_NUCLEAR[1]})

# B16 solar - sanity check
call("A75 B16 solar (Jan 2023)", {**GEN_BASE, "psrType": "B16",
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})


# ──────────────────────────────────────────────────────────────────────────
section("A11 CROSS-BORDER - DE_LU zone (Jan 2023)")
# ──────────────────────────────────────────────────────────────────────────

CB_BASE = {"documentType": "A11"}

# FR
call("A11 FR→DELU (Jan 2023)", {**CB_BASE,
     "in_Domain": FR_EIC, "out_Domain": DE_LU,
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})
call("A11 DELU→FR (Jan 2023)", {**CB_BASE,
     "in_Domain": DE_LU, "out_Domain": FR_EIC,
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})

# AT
call("A11 AT→DELU (Jan 2023)", {**CB_BASE,
     "in_Domain": AT_EIC, "out_Domain": DE_LU,
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})
call("A11 DELU→AT (Jan 2023)", {**CB_BASE,
     "in_Domain": DE_LU, "out_Domain": AT_EIC,
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})

# CH
call("A11 CH→DELU (Jan 2023)", {**CB_BASE,
     "in_Domain": CH_EIC, "out_Domain": DE_LU,
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})
call("A11 DELU→CH (Jan 2023)", {**CB_BASE,
     "in_Domain": DE_LU, "out_Domain": CH_EIC,
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})


# ──────────────────────────────────────────────────────────────────────────
section("A11 CROSS-BORDER - DE_AT_LU zone (expect all EMPTY)")
# ──────────────────────────────────────────────────────────────────────────

call("A11 FR→DEATLU (Jan 2023, expect EMPTY)", {**CB_BASE,
     "in_Domain": FR_EIC, "out_Domain": DE_AT_LU,
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})
call("A11 AT→DEATLU (Jan 2023, expect EMPTY)", {**CB_BASE,
     "in_Domain": AT_EIC, "out_Domain": DE_AT_LU,
     "periodStart": W_RECENT[0], "periodEnd": W_RECENT[1]})


# ──────────────────────────────────────────────────────────────────────────
section("FR CROSS-BORDER DEEP DIVE - why 84% NaN?")
# ──────────────────────────────────────────────────────────────────────────

# Test with a one-week window to see resolution
call("A11 FR→DELU (1 week, Jan 2023 - check resolution)", {**CB_BASE,
     "in_Domain": FR_EIC, "out_Domain": DE_LU,
     "periodStart": "202301010000", "periodEnd": "202301080000"})

call("A11 DELU→FR (1 week, Jan 2023)", {**CB_BASE,
     "in_Domain": DE_LU, "out_Domain": FR_EIC,
     "periodStart": "202301010000", "periodEnd": "202301080000"})

# Also test Jan 2020 - early in dataset
call("A11 FR→DELU (Jan 2020)", {**CB_BASE,
     "in_Domain": FR_EIC, "out_Domain": DE_LU,
     "periodStart": W_OLD[0], "periodEnd": W_OLD[1]})


# ──────────────────────────────────────────────────────────────────────────
# PRINT FULL REPORT
# ──────────────────────────────────────────────────────────────────────────
section("SUMMARY")
for r in RESULTS:
    report(r)

print("\n")
ok    = sum(1 for r in RESULTS if r["status"] == "OK")
empty = sum(1 for r in RESULTS if r["status"] == "EMPTY")
fail  = sum(1 for r in RESULTS if r["status"] not in ("OK", "EMPTY"))
print(f"Total: {len(RESULTS)} calls - OK={ok}  EMPTY={empty}  FAIL/ERROR={fail}")

# For the FR deep dive, print the full raw XML of the 1-week window
# so we can see the actual resolution and data structure
fr_deep = next((r for r in RESULTS if "1 week" in r["label"] and "FR→DELU" in r["label"]), None)
if fr_deep and fr_deep["status"] == "OK":
    print("\n" + "─" * 65)
    print("RAW XML PREVIEW - FR→DELU 1-week (first 2000 chars):")
    print("─" * 65)
    # Re-fetch to get full body (we only stored preview earlier)
    p = {
        "documentType": "A11",
        "in_Domain": FR_EIC,
        "out_Domain": DE_LU,
        "periodStart": "202301010000",
        "periodEnd": "202301080000",
        "securityToken": TOKEN,
    }
    try:
        r2 = requests.get(BASE_URL, params=p, timeout=60)
        print(r2.text[:2000])
    except Exception as e:
        print(f"  Error re-fetching: {e}")
