"""Backtesting helpers for ETF proxy duration allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_TO_ETF = {
    "Underweight duration": "SHY",
    "Neutral duration": "IEF",
    "Overweight duration": "TLT",
}


def run_backtest(
    monthly_returns: pd.DataFrame,
    decision_regime: pd.Series,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Compare static Treasury ETF exposures with the lagged timing allocation.

    Transaction costs are charged only when the timing model changes ETF proxy.
    Costs are entered in basis points per switch and subtracted from that month's
    timing return.
    """

    aligned_regime = decision_regime.reindex(monthly_returns.index).ffill()
    chosen_etf = aligned_regime.map(REGIME_TO_ETF).fillna("IEF")

    timing_returns = pd.Series(index=monthly_returns.index, dtype=float, name="Timing Model")
    for etf in REGIME_TO_ETF.values():
        mask = chosen_etf == etf
        if etf in monthly_returns.columns:
            timing_returns.loc[mask] = monthly_returns.loc[mask, etf]

    switches = chosen_etf.ne(chosen_etf.shift(1)).fillna(False)
    if not switches.empty:
        switches.iloc[0] = False
    cost = switches.astype(float) * (transaction_cost_bps / 10000.0)
    timing_returns = timing_returns - cost

    results = pd.DataFrame(index=monthly_returns.index)
    if "SHY" in monthly_returns.columns:
        results["Static Short"] = monthly_returns["SHY"]
    if "IEF" in monthly_returns.columns:
        results["Static Intermediate"] = monthly_returns["IEF"]
    if "TLT" in monthly_returns.columns:
        results["Static Long"] = monthly_returns["TLT"]

    available = [col for col in ["SHY", "IEF", "TLT"] if col in monthly_returns.columns]
    if available:
        results["Equal Weight Treasury ETFs"] = monthly_returns[available].mean(axis=1)
    if {"SHY", "TLT"}.issubset(monthly_returns.columns):
        results["60/40 Short/Long"] = monthly_returns["SHY"] * 0.60 + monthly_returns["TLT"] * 0.40

    results["Timing Model"] = timing_returns
    results = results.dropna(subset=["Timing Model"])
    results["Timing ETF"] = chosen_etf.reindex(results.index)
    results["Switch"] = switches.reindex(results.index).fillna(False)
    results["Transaction Cost"] = cost.reindex(results.index).fillna(0.0)
    return results


def performance_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate common monthly-return performance statistics."""

    numeric = returns.select_dtypes(include=[np.number]).drop(columns=["Transaction Cost"], errors="ignore")
    periods_per_year = 12
    stats = {}

    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue

        growth = (1.0 + series).prod()
        years = len(series) / periods_per_year
        cagr = growth ** (1.0 / years) - 1.0 if years > 0 else np.nan
        vol = series.std(ddof=0) * np.sqrt(periods_per_year)
        sharpe = cagr / vol if vol and not np.isnan(vol) else np.nan
        max_dd = max_drawdown(series)

        stats[col] = {
            "CAGR": cagr,
            "Volatility": vol,
            "Max Drawdown": max_dd,
            "Sharpe Ratio": sharpe,
        }

    return pd.DataFrame(stats).T


def max_drawdown(returns: pd.Series) -> float:
    """Return the maximum peak-to-trough drawdown of a return series."""

    wealth = (1.0 + returns.dropna()).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return drawdown.min()


def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Convert simple returns into cumulative growth of one dollar."""

    numeric = returns.select_dtypes(include=[np.number]).drop(columns=["Transaction Cost"], errors="ignore")
    return (1.0 + numeric).cumprod()


def rolling_returns(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Calculate rolling compounded returns."""

    numeric = returns.select_dtypes(include=[np.number]).drop(columns=["Transaction Cost"], errors="ignore")
    return (1.0 + numeric).rolling(window).apply(np.prod, raw=True) - 1.0


def calendar_year_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compound monthly strategy returns into calendar-year returns."""

    numeric = returns.select_dtypes(include=[np.number]).drop(columns=["Transaction Cost"], errors="ignore")
    annual = (1.0 + numeric).resample("YE").prod() - 1.0
    annual.index = annual.index.year
    annual.index.name = "Year"
    return annual


def allocation_diagnostics(backtest_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize ETF allocation shares, turnover, and holding periods."""

    if backtest_results.empty or "Timing ETF" not in backtest_results:
        return pd.DataFrame()

    etf = backtest_results["Timing ETF"].dropna()
    switches = etf.ne(etf.shift(1)).fillna(False)
    if not switches.empty:
        switches.iloc[0] = False

    run_id = switches.cumsum()
    holding_periods = etf.groupby(run_id).size()
    months = len(etf)
    years = months / 12.0 if months else np.nan

    rows = [
        {"Metric": "Months tested", "Value": months},
        {"Metric": "Allocation changes", "Value": int(switches.sum())},
        {"Metric": "Changes per year", "Value": switches.sum() / years if years else np.nan},
        {"Metric": "Average holding period (months)", "Value": holding_periods.mean()},
        {"Metric": "Median holding period (months)", "Value": holding_periods.median()},
    ]

    for label, etf_name in [("Time in SHY", "SHY"), ("Time in IEF", "IEF"), ("Time in TLT", "TLT")]:
        rows.append({"Metric": label, "Value": (etf == etf_name).mean()})

    if "Transaction Cost" in backtest_results:
        rows.append({"Metric": "Total transaction cost drag", "Value": backtest_results["Transaction Cost"].sum()})

    return pd.DataFrame(rows)


def regime_forward_returns(monthly_returns: pd.DataFrame, decision_regime: pd.Series) -> pd.DataFrame:
    """Compare next-month ETF returns after each model regime."""

    regimes = decision_regime.reindex(monthly_returns.index).ffill()
    next_returns = monthly_returns[[col for col in ["SHY", "IEF", "TLT"] if col in monthly_returns.columns]]
    rows = []

    for regime in ["Underweight duration", "Neutral duration", "Overweight duration"]:
        mask = regimes == regime
        sample = next_returns.loc[mask]
        if sample.empty:
            continue
        selected = REGIME_TO_ETF[regime]
        for etf in next_returns.columns:
            rows.append(
                {
                    "Regime": regime,
                    "ETF": etf,
                    "Average Next-Month Return": sample[etf].mean(),
                    "Hit Rate": (sample[etf] > 0).mean(),
                    "Months": sample[etf].count(),
                    "Selected ETF": etf == selected,
                }
            )

    return pd.DataFrame(rows)


def rate_environment_returns(
    strategy_returns: pd.DataFrame,
    ten_year_yield: pd.Series,
    lookback: int = 12,
) -> pd.DataFrame:
    """Summarize performance when the 10-year yield is rising or falling."""

    numeric = strategy_returns.select_dtypes(include=[np.number]).drop(columns=["Transaction Cost"], errors="ignore")
    yield_change = ten_year_yield.reindex(numeric.index).diff(lookback)
    environments = pd.Series(index=numeric.index, dtype="object")
    environments.loc[yield_change > 0] = "Rising 10Y yield"
    environments.loc[yield_change < 0] = "Falling 10Y yield"

    rows = []
    for environment in ["Rising 10Y yield", "Falling 10Y yield"]:
        mask = environments == environment
        for col in numeric.columns:
            sample = numeric.loc[mask, col].dropna()
            if sample.empty:
                continue
            rows.append(
                {
                    "Environment": environment,
                    "Strategy": col,
                    "Average Monthly Return": sample.mean(),
                    "Annualized Return": (1.0 + sample.mean()) ** 12 - 1.0,
                    "Hit Rate": (sample > 0).mean(),
                    "Months": len(sample),
                }
            )

    return pd.DataFrame(rows)