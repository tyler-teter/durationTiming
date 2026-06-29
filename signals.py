"""Duration timing signal construction.

The signals are intentionally simple and transparent. They are inspired by
style-premia concepts commonly used in fixed income research:

* Carry: term spread / slope.
* Momentum: falling yields are favorable for long duration, so the sign is the
  negative change in the 10-year yield.
* Value: high yields or high real yields versus recent history favor extending.
* Bond risk premium proxy: long yield minus a short yield.

All rolling z-scores use trailing observations only. The final decision score
is also lagged by one month before backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SIGNAL_NAMES = ["Carry", "Momentum", "Value", "Risk Premium Proxy"]


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Compute a trailing rolling z-score without using future observations."""

    min_obs = min_periods or max(12, window // 3)
    mean = series.rolling(window=window, min_periods=min_obs).mean()
    std = series.rolling(window=window, min_periods=min_obs).std(ddof=0)
    z = (series - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def normalize_weights(weights: dict[str, float], signal_names: list[str] | None = None) -> pd.Series:
    """Normalize user-entered weights onto the signal set used by the model."""

    names = signal_names or SIGNAL_NAMES
    weight_series = pd.Series(weights, dtype=float)
    aligned = pd.Series({name: weight_series.get(name, 0.0) for name in names}, dtype=float)
    weight_sum = aligned.abs().sum()
    if weight_sum == 0:
        return aligned * np.nan
    return aligned / weight_sum


def build_raw_signals(
    yields: pd.DataFrame,
    carry_choice: str,
    momentum_lookback: int,
    value_choice: str,
) -> pd.DataFrame:
    """Create raw, economically signed duration timing signals."""

    signals = pd.DataFrame(index=yields.index)

    if carry_choice == "10Y-3M":
        signals["Carry"] = yields["10Y Treasury"] - yields["3M Treasury"]
        brp_short_rate = yields["3M Treasury"]
    else:
        signals["Carry"] = yields["10Y Treasury"] - yields["2Y Treasury"]
        brp_short_rate = yields["2Y Treasury"]

    # A decline in yields generally helps longer-duration bonds. Multiplying by
    # -1 makes a positive momentum value pro-duration.
    signals["Momentum"] = -yields["10Y Treasury"].diff(momentum_lookback)

    if value_choice == "Real yield proxy" and "10Y Real Yield Proxy" in yields.columns:
        signals["Value"] = yields["10Y Real Yield Proxy"]
    else:
        signals["Value"] = yields["10Y Treasury"]

    # This is a practical proxy, not the survey-expectations BRP estimate used
    # in richer academic implementations.
    signals["Risk Premium Proxy"] = yields["10Y Treasury"] - brp_short_rate

    return signals


def build_duration_signals(
    yields: pd.DataFrame,
    rolling_window: int,
    momentum_lookback: int,
    carry_choice: str,
    weights: dict[str, float],
    threshold: float,
    value_choice: str = "Nominal 10Y yield",
) -> pd.DataFrame:
    """Build z-scored signals, combined score, and lagged regime decisions."""

    raw = build_raw_signals(
        yields=yields,
        carry_choice=carry_choice,
        momentum_lookback=momentum_lookback,
        value_choice=value_choice,
    )

    zscores = raw.apply(lambda col: rolling_zscore(col, rolling_window))
    zscores = zscores.rename(columns={col: f"{col} Z" for col in zscores.columns})

    normalized_weights = normalize_weights(weights, [col.replace(" Z", "") for col in zscores.columns])

    if normalized_weights.isna().all():
        combined = pd.Series(np.nan, index=zscores.index, name="Duration Score")
    else:
        combined = sum(
            zscores[f"{name} Z"] * normalized_weights[name]
            for name in normalized_weights.index
        )
        combined.name = "Duration Score"

    result = pd.concat([raw, zscores, combined], axis=1)
    result["Regime"] = classify_scores(result["Duration Score"], threshold)

    # The decision score is the one available for an allocation made at the
    # next rebalance. Backtests should use this lagged value, not the same-month
    # combined score.
    result["Decision Score"] = result["Duration Score"].shift(1)
    result["Decision Regime"] = classify_scores(result["Decision Score"], threshold)

    return result


def signal_contributions(
    signals: pd.DataFrame,
    weights: dict[str, float],
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return each signal's contribution to the combined score for a date."""

    z_cols = [col for col in signals.columns if col.endswith(" Z")]
    if not z_cols:
        return pd.DataFrame()

    available = signals.dropna(subset=z_cols, how="all")
    if as_of is not None:
        available = available.loc[available.index <= pd.Timestamp(as_of)]
    if available.empty:
        return pd.DataFrame()

    selected = available.tail(1)
    names = [col.replace(" Z", "") for col in z_cols]
    normalized = normalize_weights(weights, names)
    rows = []
    for name in names:
        z_value = selected[f"{name} Z"].iloc[0]
        weight = normalized.get(name, np.nan)
        contribution = z_value * weight if pd.notna(z_value) and pd.notna(weight) else np.nan
        rows.append(
            {
                "Date": selected.index[0],
                "Signal": name,
                "Z-Score": z_value,
                "Normalized Weight": weight,
                "Score Contribution": contribution,
            }
        )

    return pd.DataFrame(rows)


def distance_to_regime_change(score: float, threshold: float) -> pd.DataFrame:
    """Show how far the score is from the next regime boundary."""

    if pd.isna(score):
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {"Boundary": "Overweight above", "Score Needed": threshold, "Distance": threshold - score},
            {"Boundary": "Underweight below", "Score Needed": -threshold, "Distance": score + threshold},
        ]
    )


def classify_scores(score: pd.Series, threshold: float) -> pd.Series:
    """Map a continuous score into duration positioning regimes."""

    regime = pd.Series("Neutral duration", index=score.index, dtype="object")
    regime = regime.mask(score > threshold, "Overweight duration")
    regime = regime.mask(score < -threshold, "Underweight duration")
    regime = regime.mask(score.isna(), np.nan)
    return regime