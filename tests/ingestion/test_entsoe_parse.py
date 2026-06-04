"""Unit tests for ENTSO-E XML parsing (``ingestion.entsoe.parse_entsoe_xml``).

These feed hand-written, minimal ENTSO-E documents (A44 price / A65 load /
A75 generation) to ``parse_entsoe_xml`` and assert on the concrete output:
the recovered timestamps, values, and that the namespace auto-detection
(``root.tag.split('}')[0]``) works for at least two different root
tags / namespace URIs.

No network is ever touched here - every input is a raw string built inline.
"""

import pandas as pd
import pytest

from ingestion.entsoe import parse_entsoe_xml


# --- Namespace URIs actually used by ENTSO-E (mirrors entsoe.NS) -----------
PRICE_NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"
LOAD_NS = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"


def _a44_price_doc():
    """Minimal A44 day-ahead price document, one TimeSeries / Period, PT60M.

    Three hourly points starting 2024-01-01T00:00Z. ``price.amount`` carries
    the value; there is no ``MktPSRType`` so no ``psr_type`` column is added.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="{PRICE_NS}">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2024-01-01T00:00Z</start>
        <end>2024-01-01T03:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>50.10</price.amount></Point>
      <Point><position>2</position><price.amount>48.25</price.amount></Point>
      <Point><position>3</position><price.amount>61.00</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""


def _a65_load_doc():
    """Minimal A65 actual-load document, one TimeSeries / Period, PT60M.

    Uses ``quantity`` (not ``price.amount``) for the value, exercising the
    fallback branch in the parser. Two hourly points from 2024-06-01T00:00Z.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="{LOAD_NS}">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2024-06-01T00:00Z</start>
        <end>2024-06-01T02:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>42000</quantity></Point>
      <Point><position>2</position><quantity>41500</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""


def _a75_generation_doc():
    """Minimal A75 generation document with two PSR-typed TimeSeries.

    Each TimeSeries carries a ``MktPSRType/psrType`` so the parser emits a
    ``psr_type`` column. One PT60M point each, same timestamp -> distinct
    rows keyed by psr_type.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="{LOAD_NS}">
  <TimeSeries>
    <MktPSRType><psrType>B16</psrType></MktPSRType>
    <Period>
      <timeInterval>
        <start>2024-03-10T00:00Z</start>
        <end>2024-03-10T01:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>1234</quantity></Point>
    </Period>
  </TimeSeries>
  <TimeSeries>
    <MktPSRType><psrType>B19</psrType></MktPSRType>
    <Period>
      <timeInterval>
        <start>2024-03-10T00:00Z</start>
        <end>2024-03-10T01:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>5678</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""


def _a65_quarter_hour_doc():
    """A65-style document at PT15M resolution to check the 15-minute stride."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="{LOAD_NS}">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2024-01-01T00:00Z</start>
        <end>2024-01-01T01:00Z</end>
      </timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>100</quantity></Point>
      <Point><position>2</position><quantity>110</quantity></Point>
      <Point><position>3</position><quantity>120</quantity></Point>
      <Point><position>4</position><quantity>130</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""


# --------------------------------------------------------------------------
# A44 price document
# --------------------------------------------------------------------------
def test_parse_a44_price_timestamps_and_values():
    df = parse_entsoe_xml(_a44_price_doc())

    assert list(df.columns) == ["datetime_utc", "value"]
    assert "psr_type" not in df.columns  # no MktPSRType -> no psr_type column
    assert len(df) == 3

    expected_ts = pd.to_datetime(
        ["2024-01-01T00:00Z", "2024-01-01T01:00Z", "2024-01-01T02:00Z"],
        utc=True,
    )
    assert list(df["datetime_utc"]) == list(expected_ts)
    assert df["value"].tolist() == [50.10, 48.25, 61.00]

    # tz-aware UTC, as documented by parse_entsoe_xml (tz_convert("UTC"))
    assert str(df["datetime_utc"].dt.tz) == "UTC"


def test_parse_a44_hourly_stride_is_60min():
    df = parse_entsoe_xml(_a44_price_doc()).sort_values("datetime_utc")
    deltas = df["datetime_utc"].diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(hours=1)


# --------------------------------------------------------------------------
# A65 load document (quantity fallback)
# --------------------------------------------------------------------------
def test_parse_a65_load_uses_quantity_fallback():
    df = parse_entsoe_xml(_a65_load_doc())

    assert list(df.columns) == ["datetime_utc", "value"]
    assert len(df) == 2

    expected_ts = pd.to_datetime(
        ["2024-06-01T00:00Z", "2024-06-01T01:00Z"], utc=True
    )
    assert list(df["datetime_utc"]) == list(expected_ts)
    # quantity branch parsed as float
    assert df["value"].tolist() == [42000.0, 41500.0]


# --------------------------------------------------------------------------
# A75 generation document (psr_type column + multiple TimeSeries)
# --------------------------------------------------------------------------
def test_parse_a75_generation_emits_psr_type():
    df = parse_entsoe_xml(_a75_generation_doc())

    assert "psr_type" in df.columns
    assert set(df["psr_type"]) == {"B16", "B19"}
    assert len(df) == 2

    # Both TimeSeries share the same timestamp but differ by psr_type.
    expected_ts = pd.to_datetime("2024-03-10T00:00Z", utc=True)
    assert (df["datetime_utc"] == expected_ts).all()

    by_psr = df.set_index("psr_type")["value"].to_dict()
    assert by_psr == {"B16": 1234.0, "B19": 5678.0}


# --------------------------------------------------------------------------
# Resolution handling
# --------------------------------------------------------------------------
def test_parse_pt15m_resolution_stride_is_15min():
    df = parse_entsoe_xml(_a65_quarter_hour_doc()).sort_values("datetime_utc")
    assert len(df) == 4

    expected_ts = pd.to_datetime(
        [
            "2024-01-01T00:00Z",
            "2024-01-01T00:15Z",
            "2024-01-01T00:30Z",
            "2024-01-01T00:45Z",
        ],
        utc=True,
    )
    assert list(df["datetime_utc"]) == list(expected_ts)
    assert df["value"].tolist() == [100.0, 110.0, 120.0, 130.0]


# --------------------------------------------------------------------------
# Namespace auto-detection across two distinct root tags / namespace URIs
# --------------------------------------------------------------------------
def test_namespace_autodetection_two_distinct_namespaces():
    # A44 lives in the 451-3 publicationdocument namespace under
    # <Publication_MarketDocument>; A65/A75 in the 451-6 generationload
    # namespace under <GL_MarketDocument>. The parser detects the namespace
    # from the root tag, so both must parse correctly with no hardcoding.
    price_df = parse_entsoe_xml(_a44_price_doc())
    load_df = parse_entsoe_xml(_a65_load_doc())

    assert PRICE_NS != LOAD_NS  # genuinely two different namespaces
    assert not price_df.empty
    assert not load_df.empty
    assert price_df["value"].iloc[0] == 50.10
    assert load_df["value"].iloc[0] == 42000.0


def test_namespace_autodetection_custom_namespace_uri():
    # A fabricated-but-well-formed namespace URI to prove detection is purely
    # driven by root.tag, not a fixed allowlist.
    custom_ns = "urn:test:made-up-namespace:9:9"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="{custom_ns}">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2024-02-02T00:00Z</start>
        <end>2024-02-02T01:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>7</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""
    df = parse_entsoe_xml(xml)
    assert len(df) == 1
    assert df["value"].iloc[0] == 7.0
    assert df["datetime_utc"].iloc[0] == pd.to_datetime(
        "2024-02-02T00:00Z", utc=True
    )


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------
def test_empty_document_returns_empty_dataframe():
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="{PRICE_NS}">
</Publication_MarketDocument>"""
    df = parse_entsoe_xml(xml)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_duplicate_timestamps_are_dropped():
    # Two TimeSeries with identical timestamps and no psr_type collapse to one
    # row (drop_duplicates on datetime_utc).
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="{PRICE_NS}">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2024-01-01T00:00Z</start>
        <end>2024-01-01T01:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>10</price.amount></Point>
    </Period>
  </TimeSeries>
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2024-01-01T00:00Z</start>
        <end>2024-01-01T01:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>10</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""
    df = parse_entsoe_xml(xml)
    assert len(df) == 1


def test_non_utc_offset_start_is_converted_to_utc():
    # ENTSO-E always publishes Z, but the parser calls tz_convert("UTC");
    # feeding a +01:00 offset proves the conversion, not just truncation.
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="{PRICE_NS}">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2024-01-01T01:00+01:00</start>
        <end>2024-01-01T02:00+01:00</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>33</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""
    df = parse_entsoe_xml(xml)
    # 01:00+01:00 == 00:00Z
    assert df["datetime_utc"].iloc[0] == pd.to_datetime(
        "2024-01-01T00:00Z", utc=True
    )
