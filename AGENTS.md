# AGENTS.md — FINS3645 Project B instructions

## Project scope

This is my individual FINS3645 Part B project. The product is BridgeSignal Funds, a transparent decision-support app for comparing equity, cryptocurrency and combined funds. Part B covers portfolio construction, out-of-sample backtesting, news sentiment, sentiment integration and a Streamlit application.

This project continues my own Part A work. Reuse the verified Part A approach for cleaning, native-calendar returns, equity-calendar alignment and daily headline assembly. Part B must build on that foundation without changing the established data rules. The Part A Coverage Confidence Index (CCI) may be reused as an input to the original sentiment-fusion extension. All reused code must be copied into this Part B folder so the submission runs independently.

## Data rules

- Read `PROJECT_BRIEF.md` and the files in `context/` before suggesting changes.
- Load all raw data through `src/data_access.py`.
- Never edit, save or submit raw course data.
- Keep the sample ending on 31 December 2023.
- Use adjusted closing prices when calculating returns.
- Calculate returns separately within each ticker’s native trading calendar.
- Calculate cryptocurrency returns on the seven-day cryptocurrency calendar before matching them with equity trading dates.
- Do not combine equity and cryptocurrency price levels before calculating returns.
- Normalise dates and timezones before joining datasets.
- Preserve the original headline capitalisation, punctuation and wording for VADER.

## Portfolio and backtest rules

- Create equity-only, cryptocurrency-only and combined funds.
- Compare equal weighting, minimum variance and risk parity.
- Use a walk-forward out-of-sample backtest.
- A weight used on a live date may only use information dated before that date.
- Rebalance monthly.
- Use a 252-observation estimation window for equity and combined funds.
- Use a 365-observation estimation window for cryptocurrency funds.
- Annualise equity and combined funds using 252 trading days.
- Annualise cryptocurrency funds using 365 days.
- Keep portfolios long-only and fully invested.
- Apply a maximum weight of 25% per asset.
- Confirm that every set of portfolio weights sums to one.
- Use a zero risk-free rate and clearly state this assumption.
- Deduct a transaction cost of 10 basis points per one-way turnover and clearly state this assumption.
- Report the estimation window, rebalance frequency and first live backtest date.
- Check that the different optimisation methods produce meaningfully different weights.

## Sentiment rules

- Apply sentiment analysis only to equity headlines.
- Use VADER with a documented finance-specific lexicon extension.
- Keep headline capitalisation and punctuation when scoring sentiment.
- Build daily ticker-level sentiment before creating sector-level results.
- Give each ticker equal weight within its sector so companies with more articles do not dominate.
- Treat ticker-days without news as neutral zero and explain this decision.
- Lag sentiment by one equity trading day to prevent look-ahead bias.
- A weekend headline aligned to Monday may first be used in Tuesday’s trading decision.
- Do not claim that sentiment predicts returns unless the out-of-sample evidence supports it.
- Clearly explain that headline sentiment is a noisy measure.
- A sentiment-fusion method that underperforms is still a valid result if it is measured and critically explained.

## Innovation

- Continue the Part A Coverage Confidence Index as part of the Part B extension.
- Test a Coverage-Aware Sentiment Tilt that combines lagged ticker sentiment with the previous available sector CCI.
- Apply this tilt only to the equity part of the investment product.
- Compare the base portfolio with the sentiment-augmented portfolio.
- Report both positive and negative results honestly.
- Explain why the extension may be useful, how it is calculated and its limitations.

## Required outputs

Save the following exact files:

- `results/data/fund_returns.csv`
- `results/data/fund_weights.csv`
- `results/data/sector_sentiment_index.csv`
- `results/tables/performance_metrics.csv`

Also produce:

- A performance-metrics table across all funds and methods.
- A growth-of-$1 figure.
- A drawdown figure.
- A portfolio-weights-over-time figure.
- A Sharpe-ratio or return-versus-risk figure.
- A sector sentiment-index time-series figure.
- A before-versus-after sentiment-fusion table and figure.
- A fact sheet for each fund, including annualised return, annualised volatility, Sharpe ratio, maximum drawdown, growth of $1 and current holdings.

## Streamlit app

- The app must read precomputed files from `results/`.
- The app must not download data, score headlines or rerun portfolio optimisation.
- Allow users to compare funds.
- Provide a fact sheet for each fund.
- Allow users to set an allocation across funds.
- Display the sector sentiment analytics.
- Explain the main assumptions and limitations in simple language.
- Keep the app suitable for deployment on a basic Streamlit machine.

## Verification

- Test date alignment, return calculations, annualisation and sentiment lagging.
- Test that only past information is used at every rebalance date.
- Test that weights are long-only, do not exceed the maximum and sum to one.
- Test the neutral treatment of no-news ticker-days.
- Test that tickers are equally weighted within sectors.
- Run the complete project script and examine every generated output.
- Run the methodology tests and `scripts/check_handin.py`.
- Do not invent numerical results.
- Every number in the report must match a generated CSV file.
- Clearly identify assumptions, limitations and unsuccessful results.

## AI use

- Use AI to explain concepts, suggest draft code, review methodology, recommend tests and help identify errors.
- Treat AI-generated code and writing as drafts that require checking.
- Give file and line evidence when reviewing code.
- Keep honest prompt logs showing what AI produced and how it was checked.
- Record any incorrect, risky or unsupported AI suggestions and explain what was changed.
- Do not fabricate prompts, checks, corrections or manual work.
- Accurately acknowledge substantial AI contributions.
- I will run the code, inspect the outputs, make the final modelling decisions and write the final interpretation in my own words.
