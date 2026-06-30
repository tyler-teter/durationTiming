"""Data loading utilities for the duration timing research app.

The app uses FRED for Treasury yields, the Philadelphia Fed SPF for optional
survey-based short-rate expectations, and Yahoo Finance for ETF total return
proxies. Functions in this module keep IO separate from research logic so the
signal and backtest modules can be tested with ordinary DataFrames.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable
import io
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf


FRED_YIELD_SERIES = {
    "DGS10": "10Y Treasury",
    "DGS3MO": "3M Treasury",
    "DGS2": "2Y Treasury",
    "DGS5": "5Y Treasury",
    "DGS30": "30Y Treasury",
    "CPIAUCSL": "CPI",
}


ETF_PROXIES = {
    "SHY": "Short Treasury",
    "IEF": "Intermediate Treasury",
    "TLT": "Long Treasury",
}


SPF_MEAN_LEVEL_URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/historical-data/meanLevel.xlsx"
    "?hash=A51C49A6FCF80FE18F2CE81D903F4970&sc_lang=en"
)

SPF_EXPECTED_SHORT_RATE_COL = "SPF Expected Short Rate"


def _as_timestamp(value: str | date | pd.Timestamp) -> pd.Timestamp:
    """Normalize user dates into pandas timestamps."""

    return pd.Timestamp(value).tz_localize(None)


def load_fred_yields(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp | None = None,
    series: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Download Treasury yield and CPI data from FRED.

    FRED Treasury yields are quoted in annual percentage points. Missing
    observations occur on weekends and holidays, so the app forward-fills only
    after resampling to month-end. This avoids inventing extra daily data.
    """

    selected = list(series or FRED_YIELD_SERIES.keys())
    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end or pd.Timestamp.today())

    raw = web.DataReader(selected, "fred", start_ts, end_ts)
    raw = raw.rename(columns=FRED_YIELD_SERIES)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    return raw


def make_monthly_yields(daily_yields: pd.DataFrame) -> pd.DataFrame:
    """Convert daily FRED data into a month-end research panel."""

    monthly = daily_yields.resample("ME").last()

    # CPI is published monthly and can have sparse dates depending on the FRED
    # response. Forward-fill CPI and yields at month-end after the resample.
    monthly = monthly.ffill()
    monthly = _drop_incomplete_final_month(monthly, daily_yields.index.max())

    if "CPI" in monthly.columns:
        monthly["Inflation YoY"] = monthly["CPI"].pct_change(12) * 100.0
        monthly["10Y Real Yield Proxy"] = monthly["10Y Treasury"] - monthly["Inflation YoY"]

    return monthly.dropna(how="all")


def load_spf_expected_short_rate(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load the Philadelphia Fed SPF long-run T-bill forecast.

    The SPF ``BILL10`` series is used as a survey-based proxy for expected
    future short rates. It is quarterly; the app dates it at quarter-end and
    forward-fills to month-end so the signal uses only information available
    after the survey quarter has closed.
    """

    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end or pd.Timestamp.today())
    quarterly = _read_spf_sheet("BILL10")

    years = pd.to_numeric(quarterly["YEAR"], errors="coerce")
    quarters = pd.to_numeric(quarterly["QUARTER"], errors="coerce")
    quarterly["Date"] = [
        _spf_quarter_end(year, quarter)
        for year, quarter in zip(years, quarters)
    ]
    quarterly[SPF_EXPECTED_SHORT_RATE_COL] = pd.to_numeric(quarterly["BILL10"], errors="coerce")
    quarterly = quarterly.set_index("Date")[[SPF_EXPECTED_SHORT_RATE_COL]].dropna()

    monthly_index = pd.date_range(start_ts + pd.offsets.MonthEnd(0), end_ts + pd.offsets.MonthEnd(0), freq="ME")
    monthly = quarterly.reindex(quarterly.index.union(monthly_index)).sort_index().ffill().reindex(monthly_index)
    monthly = monthly.loc[(monthly.index >= start_ts) & (monthly.index <= end_ts + pd.offsets.MonthEnd(0))]
    return monthly.dropna(how="all")


def _spf_quarter_end(year: float, quarter: float) -> pd.Timestamp:
    """Convert SPF YEAR/QUARTER values into quarter-end timestamps."""

    if pd.isna(year) or pd.isna(quarter):
        return pd.NaT

    year_int = int(float(year))
    quarter_int = int(float(quarter))
    if quarter_int < 1 or quarter_int > 4:
        return pd.NaT

    return pd.Timestamp(year=year_int, month=quarter_int * 3, day=1) + pd.offsets.MonthEnd(0)

def add_spf_expected_short_rate(monthly_yields: pd.DataFrame) -> pd.DataFrame:
    """Join SPF expected short-rate data onto an existing monthly yield panel."""

    if monthly_yields.empty:
        return monthly_yields

    spf = load_spf_expected_short_rate(monthly_yields.index.min(), monthly_yields.index.max())
    return monthly_yields.join(spf, how="left")


def load_etf_prices(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp | None = None,
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Download adjusted ETF prices from Yahoo Finance."""

    selected = list(tickers or ETF_PROXIES.keys())
    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end or pd.Timestamp.today())

    prices = yf.download(
        selected,
        start=start_ts,
        end=end_ts,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["Close"]
    else:
        prices = prices.to_frame(name=selected[0]) if len(selected) == 1 else prices

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    return prices.dropna(how="all")


def make_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily adjusted ETF prices into month-end total returns."""

    monthly_prices = prices.resample("ME").last().ffill()
    monthly_prices = _drop_incomplete_final_month(monthly_prices, prices.index.max())
    return monthly_prices.pct_change().dropna(how="all")


def _read_spf_sheet(sheet_name: str) -> pd.DataFrame:
    """Read an SPF workbook sheet without relying on openpyxl metadata parsing."""

    with urllib.request.urlopen(SPF_MEAN_LEVEL_URL, timeout=30) as response:
        workbook_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as workbook:
        ns = {
            "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        shared_strings = _xlsx_shared_strings(workbook, ns)
        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        sheet_names = [sheet.attrib["name"] for sheet in workbook_root.findall("a:sheets/a:sheet", ns)]
        sheet_number = sheet_names.index(sheet_name) + 1
        sheet_root = ET.fromstring(workbook.read(f"xl/worksheets/sheet{sheet_number}.xml"))

        rows = []
        for row in sheet_root.findall("a:sheetData/a:row", ns):
            values = []
            for cell in row.findall("a:c", ns):
                value_node = cell.find("a:v", ns)
                if value_node is None:
                    values.append("")
                    continue
                value = value_node.text
                if cell.attrib.get("t") == "s":
                    value = shared_strings[int(value)]
                values.append(value)
            rows.append(values)

    header = rows[0]
    return pd.DataFrame(rows[1:], columns=header)


def _xlsx_shared_strings(workbook: zipfile.ZipFile, ns: dict[str, str]) -> list[str]:
    """Extract shared strings from an XLSX archive."""

    shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings = []
    for item in shared_root.findall("a:si", ns):
        text_parts = [node.text or "" for node in item.findall(".//a:t", ns)]
        strings.append("".join(text_parts))
    return strings


def _drop_incomplete_final_month(monthly: pd.DataFrame, last_observation: pd.Timestamp) -> pd.DataFrame:
    """Remove the final resampled row when it represents a partial month."""

    if monthly.empty or pd.isna(last_observation):
        return monthly

    final_month_end = last_observation + pd.offsets.MonthEnd(0)
    if monthly.index[-1] == final_month_end and last_observation.normalize() < final_month_end.normalize():
        return monthly.iloc[:-1]

    return monthly