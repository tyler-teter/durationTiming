# Fixed Income Duration Timing Research

Built by Tyler Teter, CFP®, CFA | [LinkedIn](https://www.linkedin.com/in/tylerteter/)

This Streamlit app is a fixed income research sandbox for exploring whether style-premia-inspired signals would have favored extending, staying neutral, or shortening Treasury duration through time.

The model pulls historical Treasury yield data, converts several duration timing signals into rolling z-scores, combines them into a single duration score, classifies the score into duration regimes, and compares a timing allocation against Treasury ETF benchmarks.

## Important Disclosure

This application is provided for educational and informational purposes only and does not constitute investment or financial advice, or a recommendation to buy or sell any security. Results are based on historical third-party data and statistical models that may contain errors, assumptions, delays, or omissions. Past performance and modeled results do not guarantee future outcomes. You are solely responsible for your investment decisions and should consult a qualified professional before acting on this information.

## Project Structure

```text
app.py            Streamlit user interface
data.py           FRED and Yahoo Finance data loading
signals.py        Raw signals, z-scores, score classification, contribution logic
backtest.py       ETF proxy backtest, performance stats, diagnostics
charts.py         Plotly chart builders
requirements.txt  Python dependencies
```

## Data Sources

The app uses FRED for Treasury yields:

- `DGS10`: 10-year Treasury yield
- `DGS3MO`: 3-month Treasury yield
- `DGS2`: 2-year Treasury yield
- `DGS5`: 5-year Treasury yield
- `DGS30`: 30-year Treasury yield
- `CPIAUCSL`: CPI, used for the optional real-yield proxy

The backtest uses Yahoo Finance ETF proxies:

- `SHY`: short Treasury exposure
- `IEF`: intermediate Treasury exposure
- `TLT`: long Treasury exposure

These ETF proxies are convenient research instruments, not perfect bond index histories.

The optional SPF-based bond risk premium signal uses the Philadelphia Fed Survey of Professional Forecasters mean-level workbook:

- `BILL10`: long-run T-bill forecast, used as a survey-based expected short-rate proxy

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Model Overview

The app builds four style-premia-inspired duration timing signals.

### Carry

Carry is measured as either:

```text
10Y Treasury yield - 3M Treasury yield
```

or:

```text
10Y Treasury yield - 2Y Treasury yield
```

A high carry z-score means the yield curve is steeper than usual, which the model treats as more favorable for extending duration. A low carry z-score means the curve is flat or inverted versus history.

### Momentum

Momentum is based on the negative change in the 10-year Treasury yield:

```text
Momentum = - change in 10Y yield
```

Because bond prices generally rise when yields fall, falling yields create a positive momentum signal. A high momentum z-score favors longer duration; a low momentum z-score suggests yields have been rising more than usual.

### Value

Value uses either the nominal 10-year Treasury yield or a simple real-yield proxy:

```text
10Y real yield proxy = 10Y Treasury yield - year-over-year CPI inflation
```

A high value z-score means yields are high versus their own rolling history, which the model treats as better compensation for owning duration.

### Risk Premium Proxy

The app supports two risk premium methods.

The simple proxy uses:

```text
10Y Treasury yield - short Treasury yield
```

This is a practical proxy, but it can overlap heavily with carry.

The SPF-based proxy uses:

```text
10Y Treasury yield - SPF expected short-rate proxy
```

The SPF expected short-rate proxy comes from the Philadelphia Fed Survey of Professional Forecasters `BILL10` series. The data is quarterly, dated at quarter-end, and forward-filled to month-end. A high z-score suggests unusually high compensation for bearing longer-duration Treasury risk.

## Rolling Z-Scores

Each raw signal is converted into a rolling z-score:

```text
z-score = (current signal - rolling average) / rolling standard deviation
```

The rolling window defaults to 120 months, or 10 years. This asks:

```text
Relative to the last 10 years, is this signal high, normal, or low?
```

Higher z-scores are generally signed to mean more favorable for longer duration.

## Combined Duration Score

The app combines the signal z-scores using the sidebar weights:

```text
duration score = weighted average of signal z-scores
```

Equal weights are the natural baseline:

```text
Carry:              25%
Momentum:           25%
Value:              25%
Risk Premium Proxy: 25%
```

The threshold creates two cutoffs around zero. For example, with a `0.50` threshold:

```text
Score above +0.50      Overweight duration
Score from -0.50 to +0.50  Neutral duration
Score below -0.50      Underweight duration
```

The final decision score is lagged by one month before being used in the backtest to reduce look-ahead bias.

## Backtest Logic

The timing model rotates among ETF proxies based on the lagged duration regime:

```text
Underweight duration  SHY
Neutral duration      IEF
Overweight duration   TLT
```

The app compares the timing model against:

- `Static Short`: always invested in `SHY`
- `Static Intermediate`: always invested in `IEF`
- `Static Long`: always invested in `TLT`
- `Equal Weight Treasury ETFs`: one-third `SHY`, one-third `IEF`, one-third `TLT`
- `60/40 Short/Long`: 60% `SHY`, 40% `TLT`

The app reports:

- CAGR
- Volatility
- Max drawdown
- Sharpe ratio
- Rolling 12-month returns
- Calendar-year returns
- Allocation changes
- Changes per year
- Average and median holding period
- Time spent in each ETF proxy
- Transaction cost drag

## Robustness Tools

The app includes several research diagnostics:

- Signal contribution chart
- Distance to regime boundary
- Sensitivity heatmaps across rolling windows and thresholds
- Signal z-score versus future return scatter plots
- Performance by rising-rate and falling-rate environments
- Regime forward-return summaries

These tools help test whether the model is robust or overly dependent on one parameter setting.

## Inspiration

This project was inspired by Newfound Research's article:

[Duration Timing with Style Premia](https://blog.thinknewfound.com/2017/06/duration-timing-style-premia/)

The implementation here is a simplified educational research model, not a replication of Newfound's full methodology.
