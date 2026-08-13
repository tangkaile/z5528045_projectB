# AI Use Notes — FINS3645 Project B

## How I used AI

I used AI as an assistant throughout Project B. It helped me understand parts of the project brief, suggest possible code structures, troubleshoot errors, develop tests, improve the Streamlit interface and make parts of my report clearer.

I completed the work in PyCharm and checked the suggestions before using them. I ran the pipeline and tests myself, reviewed the generated tables and figures, tested each page of the Streamlit application and made the final decisions about the methods and interpretation of the results.

## Main tasks where AI assisted

AI assisted me with:

- understanding the Part B requirements and how they continued from Part A;
- structuring the equity, cryptocurrency and combined fund families;
- implementing Equal Weight, Minimum Variance and Risk Parity portfolios;
- setting up rolling out-of-sample backtests with monthly rebalancing;
- applying the 252-observation equity window and 365-observation cryptocurrency window;
- adding the 25% maximum asset weight and long-only constraints;
- calculating turnover and applying a 10-basis-point transaction cost;
- implementing VADER headline sentiment while preserving punctuation and capitalisation;
- applying a one-equity-trading-day delay to the sentiment signal;
- developing the Coverage-Aware Sentiment Tilt (CAST) innovation;
- suggesting methodology tests and helping diagnose failed tests;
- creating figures, result tables and the Streamlit dashboard;
- improving the organisation and clarity of some report sections.

## How I checked the output

I did not accept the AI output automatically. I manually checked that:

- portfolio weights totalled 100%;
- individual asset weights did not exceed 25%;
- the portfolios were long-only and fully invested;
- equity and combined backtests used a 252-observation estimation window;
- cryptocurrency backtests used a 365-observation estimation window;
- only information available before each rebalance date was used;
- portfolios were rebalanced monthly;
- cryptocurrency returns were calculated before alignment with equity trading dates;
- headline sentiment was delayed by one equity trading day;
- days without news were treated as neutral;
- transaction costs were deducted from the live portfolio returns;
- the CAST signal used lagged sentiment and the previous available Coverage Confidence Index;
- figures and report values matched the generated CSV files;
- all nine methodology tests passed;
- all 21 mechanical hand-in checks passed;
- the Compare Funds, Fact Sheet, Allocation Lab, Sentiment Lens and Methodology pages worked correctly.

## Problems identified in AI suggestions

Some AI suggestions were incomplete or incorrect and required changes.

First, an early test command passed a `family` argument to `oos_backtest()`, but the function did not originally accept that argument. I checked the function interface and corrected the mismatch before continuing.

Second, some Python files were initially created without the `.py` extension. This caused an import error for `src.visuals`. I renamed the files correctly and tested the imports again.

Third, a terminal command was accidentally inserted into a sample sentiment headline. This changed the test text and produced an unreliable result. I corrected the headline and reran the sentiment check.

Fourth, successfully running the code did not prove that the backtest was methodologically correct. I therefore added tests for estimation windows, annualisation, portfolio constraints, transaction costs, return calculations and maximum drawdown.

Finally, the initial discussion of CAST could have made the innovation sound more successful than the results showed. After checking the generated performance table, I reported that CAST slightly reduced maximum drawdown but produced a lower return and Sharpe ratio and higher turnover than the base Equity Minimum Variance portfolio.

## Changes I made

I made the following decisions and corrections after reviewing the AI suggestions:

- kept the Part A data-cleaning and calendar rules;
- used separate annualisation values for equities and cryptocurrencies;
- applied monthly out-of-sample rebalancing;
- added covariance shrinkage to the Minimum Variance method;
- added portfolio weight and long-only controls;
- preserved the original headline wording for VADER;
- used equal ticker weighting when creating sector sentiment;
- delayed sentiment by one equity trading day;
- used lagged coverage confidence in CAST to avoid using future information;
- reported CAST as a negative performance result instead of claiming that it improved the portfolio;
- revised report wording so that it matched the generated evidence;
- tested the complete Streamlit investor journey before submission.

## What I learned

The main lesson from using AI was that working code is not enough to show that a financial analysis is correct. The timing rules, estimation periods, trading calendars, portfolio constraints and transaction costs must also be checked.

I also learned that an innovation does not need to outperform to provide a useful result. CAST did not improve the main risk-adjusted performance measures in this historical sample. Reporting that result honestly helped identify the possible effect of higher turnover and showed that sentiment-based portfolio adjustments require further testing.

## Final responsibility

AI assisted with explanations, draft code, debugging, testing ideas, interface development and wording suggestions. I reviewed these suggestions, ran and checked the analysis in PyCharm, compared the report with the generated evidence and made the final methodological and interpretive decisions.
