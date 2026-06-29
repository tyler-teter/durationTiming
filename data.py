"""Data loading utilities for the duration timing research app.

The app uses FRED for Treasury yields and Yahoo Finance for ETF total return
proxies. Functions in this module keep IO separate from research logic so the
signal and backtest modules can be tested with ordinary DataFrames.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

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


def _drop_incomplete_final_month(monthly: pd.DataFrame, last_observation: pd.Timestamp) -> pd.DataFrame:
    """Remove the final resampled row when it represents a partial month."""

    if monthly.empty or pd.isna(last_observation):
        return monthly

    final_month_end = last_observation + pd.offsets.MonthEnd(0)
    if monthly.index[-1] == final_month_end and last_observation.normalize() < final_month_end.normalize():
        return monthly.iloc[:-1]

    return monthly