# Prompt log — Part B implementation and debugging

## What I wanted

I used AI to help develop and debug the Part B portfolio, sentiment, CAST, visualisation and Streamlit components. I wanted the implementation to follow the project brief and continue the cleaned data methods used in Part A.

## Prompt(s)

I asked AI to:

- propose Python functions for Equal Weight, Minimum Variance and Risk Parity portfolios;
- construct a monthly out-of-sample backtest using only earlier observations;
- help implement finance-extended VADER sentiment with a one-trading-day delay;
- combine lagged sentiment with the previous Coverage Confidence Index in the CAST extension;
- diagnose errors shown in my PyCharm terminal;
- suggest methodology tests and checks for the generated results;
- help organise the Streamlit app and report exhibits.

Most follow-up prompts asked AI to explain the next step, identify the cause of an error or show where suggested code belonged.

## What the assistant produced

AI provided draft code structures, terminal checks and explanations for the portfolio backtest, sentiment index, CAST weight adjustment, figures and Streamlit dashboard. It also suggested tests for portfolio weights, estimation windows, return calculations, annualisation, drawdown and transaction costs.

I used the suggestions as a starting point and ran each component in PyCharm before continuing.

## What was wrong or risky

Several AI suggestions did not work correctly on the first attempt:

- The proposed `oos_backtest()` call included a `family` argument that the function did not initially accept. This caused an `unexpected keyword argument 'family'` error.
- Two Python files were initially saved without the `.py` extension. This caused file-not-found and module-import errors.
- One terminal command was accidentally inserted into a sentiment test headline, producing an incorrect test input.
- Some suggested checks only confirmed that code ran; they did not prove that the backtest was free from look-ahead bias or economically reliable.
- The CAST idea could have been described as an improvement before its results were known. The completed results showed that it did not outperform the base Equity Minimum Variance fund.

## What I changed and why

I corrected the function interface so each result retained its fund-family label. I renamed the test and visualisation files with the correct `.py` extension and corrected the sentiment test input.

I then ran the methodology tests and manually checked that:

- each portfolio’s weights summed to one;
- no asset exceeded the 25% limit;
- each live rebalance used only previous observations;
- equity and combined funds used 252-observation windows;
- cryptocurrency funds used 365-observation windows;
- sentiment was delayed by one equity trading day;
- transaction costs were deducted using turnover;
- the figures and dashboard values matched the generated CSV files.

I also reported the CAST result honestly. It slightly reduced maximum drawdown, but lowered return and Sharpe ratio and increased turnover. I therefore treated it as an experimental negative result rather than claiming that it improved the portfolio.

AI assisted with drafting, code suggestions and troubleshooting. I made the final methodological choices, ran the checks, reviewed the outputs and wrote the economic interpretation in my own words.
