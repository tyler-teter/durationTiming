"""Streamlit app for fixed income duration timing research."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from backtest import (
    allocation_diagnostics,
    calendar_year_returns,
    cumulative_returns,
    performance_stats,
    rate_environment_returns,
    regime_forward_returns,
    rolling_returns,
    run_backtest,
)
from charts import (
    allocation_chart,
    calendar_year_return_chart,
    contribution_chart,
    duration_score_chart,
    forward_return_scatter,
    performance_chart,
    regime_chart,
    rolling_return_chart,
    sensitivity_heatmap,
    signal_zscore_chart,
    yield_chart,
)
from data import load_etf_prices, load_fred_yields, make_monthly_returns, make_monthly_yields
from signals import build_duration_signals, distance_to_regime_change, signal_contributions


st.set_page_config(
    page_title="Duration Timing Research",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_yields(start: date, end: date) -> pd.DataFrame:
    """Cache FRED data during a Streamlit session."""

    return make_monthly_yields(load_fred_yields(start=start, end=end))


@st.cache_data(show_spinner=False)
def cached_etf_returns(start: date, end: date) -> pd.DataFrame:
    """Cache ETF proxy returns during a Streamlit session."""

    prices = load_etf_prices(start=start, end=end)
    return make_monthly_returns(prices)


@st.cache_data(show_spinner=False)
def run_sensitivity_grid(
    yields: pd.DataFrame,
    etf_returns: pd.DataFrame,
    weights: dict[str, float],
    carry_choice: str,
    value_choice: str,
    momentum_lookback: int,
    transaction_cost_bps: float,
    windows: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> pd.DataFrame:
    """Run a compact parameter grid for robustness checks."""

    rows = []
    for window in windows:
        for threshold_value in thresholds:
            grid_signals = build_duration_signals(
                yields=yields,
                rolling_window=window,
                momentum_lookback=momentum_lookback,
                carry_choice=carry_choice,
                weights=weights,
                threshold=threshold_value,
                value_choice=value_choice,
            )
            grid_backtest = run_backtest(
                etf_returns,
                grid_signals["Decision Regime"],
                transaction_cost_bps=transaction_cost_bps,
            )
            if grid_backtest.empty:
                continue

            strategy_returns = grid_backtest[["Timing Model"]]
            stats = performance_stats(strategy_returns)
            diagnostics = allocation_diagnostics(grid_backtest)
            changes = _diagnostic_value(diagnostics, "Allocation changes")
            years = len(grid_backtest) / 12.0
            time_in_tlt = _diagnostic_value(diagnostics, "Time in TLT")

            rows.append(
                {
                    "Rolling Window": window,
                    "Threshold": threshold_value,
                    "CAGR": stats.loc["Timing Model", "CAGR"],
                    "Sharpe Ratio": stats.loc["Timing Model", "Sharpe Ratio"],
                    "Max Drawdown": stats.loc["Timing Model", "Max Drawdown"],
                    "Changes Per Year": changes / years if years else float("nan"),
                    "Time in TLT": time_in_tlt,
                }
            )

    return pd.DataFrame(rows)


def _diagnostic_value(diagnostics: pd.DataFrame, metric: str) -> float:
    """Read a single diagnostic value from the diagnostics table."""

    if diagnostics.empty:
        return float("nan")
    match = diagnostics.loc[diagnostics["Metric"] == metric, "Value"]
    return float(match.iloc[0]) if not match.empty else float("nan")


def percent_style(df: pd.DataFrame, percent_cols: list[str], number_cols: list[str] | None = None):
    """Return a Streamlit-friendly styled table."""

    formats = {col: "{:.2%}" for col in percent_cols if col in df.columns}
    formats.update({col: "{:.2f}" for col in (number_cols or []) if col in df.columns})
    return df.style.format(formats)


st.title("Fixed Income Duration Timing Research")
st.markdown("Built by Tyler Teter, CFP®, CFA | [LinkedIn](https://www.linkedin.com/in/tylerteter/)")
st.caption(
    "A style-premia-inspired model for evaluating when signals favor extending, "
    "staying neutral, or shortening Treasury duration."
)

st.warning(
    "Important disclosure: This application is provided for educational and "
    "informational purposes only and does not constitute investment or financial "
    "advice, or a recommendation to buy or sell any security. Results are based "
    "on historical third-party data and statistical models that may contain errors, "
    "assumptions, delays, or omissions. Past performance and modeled results do "
    "not guarantee future outcomes. You are solely responsible for your investment "
    "decisions and should consult a qualified professional before acting on this "
    "information."
)

with st.expander("How the math works"):
    st.markdown(
        """
        Each raw signal is converted into a rolling z-score:

        ```text
        z-score = (current signal - rolling average) / rolling standard deviation
        ```

        A z-score tells you how unusual the current signal is versus its own recent
        history. A value of `+1.0` means one standard deviation above normal; `-1.0`
        means one standard deviation below normal.

        The app then combines the signal z-scores using the sidebar weights:

        ```text
        duration score = weighted average of signal z-scores
        ```

        The threshold creates two cutoffs around zero. With a `0.50` threshold,
        scores above `+0.50` are overweight duration, scores below `-0.50` are
        underweight duration, and scores from `-0.50` to `+0.50` are neutral.

        For background, see [standard score / z-score math](https://en.wikipedia.org/wiki/Standard_score)
        and the Newfound Research article that inspired this model:
        [Duration Timing with Style Premia](https://blog.thinknewfound.com/2017/06/duration-timing-style-premia/).
        """
    )

with st.sidebar:
    st.header("Research Settings")
    start_date = st.date_input("Start date", value=date(2003, 1, 1), min_value=date(1962, 1, 1))
    end_date = st.date_input("End date", value=date.today(), min_value=start_date)

    rolling_window = st.slider("Rolling z-score window (months)", 24, 180, 120, 12)
    momentum_lookback = st.slider("Momentum lookback (months)", 3, 24, 12, 1)
    carry_choice = st.radio("Carry spread", ["10Y-3M", "10Y-2Y"], horizontal=True)
    value_choice = st.radio("Value signal", ["Nominal 10Y yield", "Real yield proxy"])
    threshold = st.slider(
        "Overweight / underweight threshold",
        0.10,
        2.00,
        0.50,
        0.05,
        help=(
            "This one number creates two cutoffs around zero. For example, a "
            "0.50 threshold means scores above +0.50 are overweight duration, "
            "scores below -0.50 are underweight duration, and scores from -0.50 "
            "to +0.50 are neutral. Raising the threshold widens the neutral zone "
            "and makes the model trade less often."
        ),
    )
    transaction_cost_bps = st.slider(
        "Cost per allocation switch (bps)",
        0.0,
        25.0,
        2.0,
        0.5,
        help="Cost deducted whenever the timing model changes ETF proxy. Entered in basis points per switch.",
    )

    st.header("Signal Weights")
    carry_weight = st.number_input(
        "Carry",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.25,
        help=(
            "Term-spread signal using the selected 10Y-minus-short-rate spread. "
            "High z-score: the curve is steeper than usual, so the model sees "
            "more compensation for extending duration. Low z-score: the curve is "
            "flat or inverted versus history, so the model is less willing to own "
            "longer duration."
        ),
    )
    momentum_weight = st.number_input(
        "Momentum",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.25,
        help=(
            "Trend signal based on the negative change in the 10Y yield. High "
            "z-score: yields have been falling more than usual, which favors "
            "longer duration. Low z-score: yields have been rising more than usual, "
            "which favors shorter duration or a neutral stance."
        ),
    )
    value_weight = st.number_input(
        "Value",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.25,
        help=(
            "Yield valuation signal using the nominal 10Y yield or real-yield "
            "proxy. High z-score: yields are high versus their own history, so "
            "duration looks better compensated. Low z-score: yields are low versus "
            "history, so long-duration exposure looks less attractive."
        ),
    )
    brp_weight = st.number_input(
        "Risk premium proxy",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.25,
        help=(
            "Simple bond-risk-premium proxy using the 10Y yield minus a short "
            "Treasury yield. High z-score: unusually high compensation for bearing "
            "longer-duration Treasury risk. Low z-score: unusually low compensation, "
            "so the model is less interested in extending duration."
        ),
    )

weights = {
    "Carry": carry_weight,
    "Momentum": momentum_weight,
    "Value": value_weight,
    "Risk Premium Proxy": brp_weight,
}

try:
    yields = cached_yields(start_date, end_date)
except Exception as exc:
    st.error(f"Could not load FRED data: {exc}")
    st.stop()

signals = build_duration_signals(
    yields=yields,
    rolling_window=rolling_window,
    momentum_lookback=momentum_lookback,
    carry_choice=carry_choice,
    weights=weights,
    threshold=threshold,
    value_choice=value_choice,
)

latest = signals.dropna(subset=["Decision Score"]).tail(1)
if not latest.empty:
    current_regime = latest["Decision Regime"].iloc[0]
    current_score = latest["Decision Score"].iloc[0]
    current_date = latest.index[0].date()
else:
    current_regime = "Not enough history"
    current_score = float("nan")
    current_date = None

metric_cols = st.columns(4)
metric_cols[0].metric("Latest decision regime", current_regime)
metric_cols[1].metric("Lagged decision score", f"{current_score:.2f}" if pd.notna(current_score) else "n/a")
metric_cols[2].metric("Decision date", str(current_date) if current_date else "n/a")
metric_cols[3].metric("10Y yield", f"{yields['10Y Treasury'].dropna().iloc[-1]:.2f}%")

tabs = st.tabs(["Signals", "Current Drivers", "Regimes", "Backtest", "Robustness", "Data"])

with tabs[0]:
    st.plotly_chart(yield_chart(yields), use_container_width=True)
    st.plotly_chart(signal_zscore_chart(signals), use_container_width=True)
    st.plotly_chart(duration_score_chart(signals, threshold), use_container_width=True)

    display_cols = [
        "Carry Z",
        "Momentum Z",
        "Value Z",
        "Risk Premium Proxy Z",
        "Duration Score",
        "Decision Score",
        "Decision Regime",
    ]
    st.dataframe(signals[display_cols].dropna(how="all").tail(24), use_container_width=True)

with tabs[1]:
    valid_driver_rows = signals.dropna(subset=["Duration Score"])
    if valid_driver_rows.empty:
        st.info("Not enough signal history to inspect drivers.")
    else:
        selected_driver_date = st.date_input(
            "Driver date",
            value=valid_driver_rows.index[-1].date(),
            min_value=valid_driver_rows.index[0].date(),
            max_value=valid_driver_rows.index[-1].date(),
            help="Pick any date. The app will use the latest completed model month on or before that date.",
        )
        selected_timestamp = valid_driver_rows.loc[valid_driver_rows.index <= pd.Timestamp(selected_driver_date)].index[-1]
        selected_row = signals.loc[selected_timestamp]
        contributions = signal_contributions(signals, weights, as_of=selected_timestamp)

        st.caption(f"Inspecting model drivers for {selected_timestamp.date()}.")
        selected_cols = st.columns(4)
        selected_cols[0].metric(
            "Same-month score",
            f"{selected_row['Duration Score']:.2f}",
            help=(
                "What the model says after the selected month is over. Useful for "
                "understanding the drivers, but it would not have been known at "
                "the start of that month."
            ),
        )
        selected_cols[1].metric(
            "Same-month regime",
            selected_row["Regime"],
            help=(
                "The duration call after seeing the selected month's data. Think "
                "of this as the model's end-of-month read, not the trade used "
                "during that month."
            ),
        )
        selected_cols[2].metric(
            "Lagged decision score",
            f"{selected_row['Decision Score']:.2f}" if pd.notna(selected_row["Decision Score"]) else "n/a",
            help=(
                "What the model would have known before the selected month began. "
                "This is the score used for the backtest allocation."
            ),
        )
        selected_cols[3].metric(
            "Lagged decision regime",
            selected_row["Decision Regime"] if pd.notna(selected_row["Decision Regime"]) else "n/a",
            help=(
                "The actual backtest position for the selected month: SHY for "
                "underweight, IEF for neutral, or TLT for overweight."
            ),
        )

        st.plotly_chart(contribution_chart(contributions), use_container_width=True)

        score_cols = st.columns(2)
        with score_cols[0]:
            st.subheader("Selected Signal Contribution")
            if contributions.empty:
                st.info("Not enough signal history to calculate contributions.")
            else:
                st.dataframe(
                    contributions.drop(columns=["Date"], errors="ignore").style.format(
                        {
                            "Z-Score": "{:.2f}",
                            "Normalized Weight": "{:.1%}",
                            "Score Contribution": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )
        with score_cols[1]:
            st.subheader("Distance to Regime Boundaries")
            boundaries = distance_to_regime_change(selected_row["Duration Score"], threshold)
            if boundaries.empty:
                st.info("Not enough history to calculate boundary distance.")
            else:
                st.dataframe(boundaries.style.format({"Score Needed": "{:.2f}", "Distance": "{:.2f}"}), use_container_width=True)

with tabs[2]:
    st.plotly_chart(regime_chart(yields, signals), use_container_width=True)
    regime_counts = signals["Decision Regime"].value_counts(normalize=True).rename("Share of months")
    st.dataframe(regime_counts.to_frame().style.format("{:.1%}"), use_container_width=True)

with tabs[3]:
    try:
        etf_returns = cached_etf_returns(start_date, end_date)
    except Exception as exc:
        st.error(f"Could not load ETF data from Yahoo Finance: {exc}")
        st.stop()

    backtest_results = run_backtest(etf_returns, signals["Decision Regime"], transaction_cost_bps=transaction_cost_bps)
    if backtest_results.empty:
        st.warning("Not enough overlapping signal and ETF history to run the backtest.")
    else:
        strategy_cols = [
            col
            for col in backtest_results.select_dtypes(include="number").columns
            if col != "Transaction Cost"
        ]
        numeric_returns = backtest_results[strategy_cols]
        stats = performance_stats(numeric_returns)
        cumulative = cumulative_returns(numeric_returns)
        rolling_12m = rolling_returns(numeric_returns, window=12)
        calendar_returns = calendar_year_returns(numeric_returns)
        environments = rate_environment_returns(
            strategy_returns=numeric_returns,
            ten_year_yield=yields["10Y Treasury"],
            lookback=12,
        )
        diagnostics = allocation_diagnostics(backtest_results)
        regime_returns = regime_forward_returns(etf_returns, signals["Decision Regime"])

        st.plotly_chart(performance_chart(cumulative), use_container_width=True)

        with st.expander("What are these backtest benchmarks?"):
            st.markdown(
                """
                The static benchmarks are buy-and-hold Treasury ETF proxies:

                - **Static Short**: always invested in `SHY`, the short-term Treasury ETF proxy.
                - **Static Intermediate**: always invested in `IEF`, the intermediate Treasury ETF proxy.
                - **Static Long**: always invested in `TLT`, the long Treasury ETF proxy.
                - **Equal Weight Treasury ETFs**: one-third `SHY`, one-third `IEF`, and one-third `TLT`, rebalanced monthly in the simple return calculation.
                - **60/40 Short/Long**: 60% `SHY` and 40% `TLT`, a simple barbell benchmark that mixes short Treasury stability with long-duration exposure.
                - **Timing Model**: rotates among `SHY`, `IEF`, and `TLT` based on the lagged duration regime signal. Underweight uses `SHY`, neutral uses `IEF`, and overweight uses `TLT`.

                These are proxies, not perfect bond indexes. They are useful for comparing whether the timing model adds value versus simple static Treasury exposures.
                """
            )

        st.plotly_chart(allocation_chart(backtest_results), use_container_width=True)

        st.subheader("Performance Statistics")
        st.dataframe(
            stats.style.format(
                {
                    "CAGR": "{:.2%}",
                    "Volatility": "{:.2%}",
                    "Max Drawdown": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        diag_cols = st.columns(2)
        with diag_cols[0]:
            st.subheader("Allocation Diagnostics")
            percent_metrics = ["Time in SHY", "Time in IEF", "Time in TLT", "Total transaction cost drag"]
            formatted = diagnostics.copy()
            formatted["Display"] = formatted.apply(
                lambda row: f"{row['Value']:.1%}" if row["Metric"] in percent_metrics else f"{row['Value']:.2f}",
                axis=1,
            )
            st.dataframe(formatted[["Metric", "Display"]], use_container_width=True, hide_index=True)
        with diag_cols[1]:
            st.subheader("Regime Forward Returns")
            if regime_returns.empty:
                st.info("Not enough data to summarize forward returns by regime.")
            else:
                st.dataframe(
                    regime_returns.style.format(
                        {
                            "Average Next-Month Return": "{:.2%}",
                            "Hit Rate": "{:.1%}",
                            "Months": "{:.0f}",
                        }
                    ),
                    use_container_width=True,
                )

        st.plotly_chart(rolling_return_chart(rolling_12m), use_container_width=True)
        st.plotly_chart(calendar_year_return_chart(calendar_returns), use_container_width=True)

        st.subheader("Rate Environment Performance")
        if environments.empty:
            st.info("Not enough data to classify rising-rate and falling-rate periods.")
        else:
            st.dataframe(
                environments.style.format(
                    {
                        "Average Monthly Return": "{:.2%}",
                        "Annualized Return": "{:.2%}",
                        "Hit Rate": "{:.1%}",
                        "Months": "{:.0f}",
                    }
                ),
                use_container_width=True,
            )

with tabs[4]:
    try:
        etf_returns = cached_etf_returns(start_date, end_date)
    except Exception as exc:
        st.error(f"Could not load ETF data from Yahoo Finance: {exc}")
        st.stop()

    st.subheader("Sensitivity Heatmaps")
    grid_windows = (60, 90, 120, 180)
    grid_thresholds = (0.25, 0.50, 0.75, 1.00)
    sensitivity = run_sensitivity_grid(
        yields=yields,
        etf_returns=etf_returns,
        weights=weights,
        carry_choice=carry_choice,
        value_choice=value_choice,
        momentum_lookback=momentum_lookback,
        transaction_cost_bps=transaction_cost_bps,
        windows=grid_windows,
        thresholds=grid_thresholds,
    )

    if sensitivity.empty:
        st.info("Not enough overlapping history for the sensitivity grid.")
    else:
        heat_cols = st.columns(2)
        with heat_cols[0]:
            st.plotly_chart(sensitivity_heatmap(sensitivity, "Sharpe Ratio", "Sensitivity: Sharpe Ratio"), use_container_width=True)
        with heat_cols[1]:
            st.plotly_chart(
                sensitivity_heatmap(
                    sensitivity,
                    "Changes Per Year",
                    "Sensitivity: Changes Per Year",
                    color_scale="RdYlGn_r",
                ),
                use_container_width=True,
            )
        st.dataframe(
            sensitivity.style.format(
                {
                    "Threshold": "{:.2f}",
                    "CAGR": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "Changes Per Year": "{:.2f}",
                    "Time in TLT": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

    st.subheader("Signal vs Future Returns")
    future_horizon = st.selectbox("Future return horizon", [1, 3, 12], index=2)
    future_etf = st.selectbox("Future return ETF", [col for col in ["SHY", "IEF", "TLT"] if col in etf_returns.columns], index=2)
    signal_col = st.selectbox(
        "Signal z-score",
        ["Carry Z", "Momentum Z", "Value Z", "Risk Premium Proxy Z", "Duration Score"],
    )
    future_return = (1.0 + etf_returns[future_etf]).rolling(future_horizon).apply(lambda values: values.prod(), raw=True) - 1.0
    future_return = future_return.shift(-future_horizon).rename(f"Next {future_horizon}M {future_etf} Return")
    scatter_data = pd.concat([signals[signal_col], future_return], axis=1)
    st.plotly_chart(forward_return_scatter(scatter_data, signal_col, future_return.name), use_container_width=True)

with tabs[5]:
    st.subheader("Monthly FRED Panel")
    st.dataframe(yields.tail(60), use_container_width=True)

    st.subheader("Raw and Transformed Signals")
    st.dataframe(signals.tail(60), use_container_width=True)

st.caption(
    "Research note: rolling statistics are trailing-only, and the allocation "
    "backtest uses the final duration score lagged by one monthly period."
)