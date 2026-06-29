"""Plotly chart builders used by the Streamlit interface."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


REGIME_COLORS = {
    "Overweight duration": "rgba(34, 139, 34, 0.16)",
    "Neutral duration": "rgba(120, 120, 120, 0.10)",
    "Underweight duration": "rgba(178, 34, 34, 0.16)",
}


def yield_chart(yields: pd.DataFrame) -> go.Figure:
    """Plot the historical 10-year Treasury yield."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=yields.index,
            y=yields["10Y Treasury"],
            mode="lines",
            name="10Y Treasury yield",
            line=dict(color="#1f4e79", width=2),
        )
    )
    fig.update_layout(
        title="Historical 10-Year Treasury Yield",
        yaxis_title="Yield (%)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def signal_zscore_chart(signals: pd.DataFrame) -> go.Figure:
    """Plot individual signal z-scores."""

    z_cols = [col for col in signals.columns if col.endswith(" Z")]
    fig = go.Figure()
    for col in z_cols:
        fig.add_trace(go.Scatter(x=signals.index, y=signals[col], mode="lines", name=col))

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Individual Signal Z-Scores",
        yaxis_title="Z-score",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def duration_score_chart(signals: pd.DataFrame, threshold: float) -> go.Figure:
    """Plot combined duration score and decision thresholds."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals["Duration Score"],
            mode="lines",
            name="Combined score",
            line=dict(color="#2f5597", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals["Decision Score"],
            mode="lines",
            name="Lagged decision score",
            line=dict(color="#c55a11", width=1.5, dash="dot"),
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="green")
    fig.add_hline(y=-threshold, line_dash="dash", line_color="firebrick")
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Combined Duration Score",
        yaxis_title="Score",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def contribution_chart(contributions: pd.DataFrame) -> go.Figure:
    """Plot latest weighted signal contributions."""

    if contributions.empty:
        return go.Figure()

    colors = contributions["Score Contribution"].apply(lambda value: "#2e7d32" if value >= 0 else "#b22222")
    fig = go.Figure(
        go.Bar(
            x=contributions["Signal"],
            y=contributions["Score Contribution"],
            marker_color=colors,
            text=contributions["Score Contribution"].map(lambda value: f"{value:.2f}"),
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Latest Signal Contributions to Duration Score",
        yaxis_title="Contribution",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def regime_chart(yields: pd.DataFrame, signals: pd.DataFrame) -> go.Figure:
    """Overlay decision regimes on the 10-year yield."""

    fig = yield_chart(yields)
    regimes = signals["Decision Regime"].dropna()

    if regimes.empty:
        return fig

    for start, end, regime in _regime_spans(regimes):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=REGIME_COLORS.get(regime, "rgba(120,120,120,0.08)"),
            opacity=1.0,
            line_width=0,
            layer="below",
        )

    fig.update_layout(title="10-Year Yield with Lagged Duration Regimes")
    return fig


def performance_chart(cumulative: pd.DataFrame) -> go.Figure:
    """Plot cumulative strategy wealth."""

    fig = go.Figure()
    for col in cumulative.columns:
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative[col], mode="lines", name=col))

    fig.update_layout(
        title="Backtest Growth of $1",
        yaxis_title="Growth of $1",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def rolling_return_chart(rolling_returns: pd.DataFrame) -> go.Figure:
    """Plot rolling 12-month returns."""

    fig = go.Figure()
    for col in rolling_returns.columns:
        fig.add_trace(go.Scatter(x=rolling_returns.index, y=rolling_returns[col], mode="lines", name=col))

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Rolling 12-Month Returns",
        yaxis_title="Return",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def calendar_year_return_chart(calendar_returns: pd.DataFrame) -> go.Figure:
    """Plot calendar-year returns as grouped bars."""

    fig = go.Figure()
    for col in calendar_returns.columns:
        fig.add_trace(
            go.Bar(
                x=calendar_returns.index.astype(str),
                y=calendar_returns[col],
                name=col,
            )
        )

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Calendar-Year Returns",
        xaxis_title="Year",
        yaxis_title="Return",
        barmode="group",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig



def allocation_chart(backtest_results: pd.DataFrame) -> go.Figure:
    """Show which ETF proxy the timing model selected through time."""

    mapping = {"SHY": 0, "IEF": 1, "TLT": 2}
    allocation = backtest_results["Timing ETF"].map(mapping)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=allocation.index,
            y=allocation,
            mode="lines",
            line_shape="hv",
            name="Timing ETF",
            line=dict(color="#7030a0", width=2),
        )
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["SHY", "IEF", "TLT"],
        title="ETF proxy",
    )
    fig.update_layout(
        title="Timing Model Allocation",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def sensitivity_heatmap(
    results: pd.DataFrame,
    value_col: str,
    title: str,
    color_scale: str = "RdYlGn",
) -> go.Figure:
    """Plot threshold by z-score-window sensitivity results."""

    if results.empty:
        return go.Figure()

    pivot = results.pivot_table(index="Threshold", columns="Rolling Window", values=value_col)
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=color_scale,
        title=title,
    )
    fig.update_layout(template="plotly_white")
    return fig


def forward_return_scatter(data: pd.DataFrame, signal_col: str, return_col: str) -> go.Figure:
    """Plot signal z-score versus future return."""

    fig = px.scatter(
        data.dropna(subset=[signal_col, return_col]),
        x=signal_col,
        y=return_col,
        trendline="ols",
        title=f"{signal_col} vs {return_col}",
        labels={return_col: "Future return", signal_col: "Signal z-score"},
        template="plotly_white",
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
    return fig


def _regime_spans(regimes: pd.Series):
    """Yield contiguous spans of equal regime labels."""

    start = regimes.index[0]
    previous_date = regimes.index[0]
    previous_regime = regimes.iloc[0]

    for current_date, current_regime in regimes.iloc[1:].items():
        if current_regime != previous_regime:
            yield start, previous_date, previous_regime
            start = current_date
            previous_regime = current_regime
        previous_date = current_date

    yield start, previous_date, previous_regime