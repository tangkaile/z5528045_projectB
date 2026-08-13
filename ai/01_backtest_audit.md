# Prompt log — Part B backtest design audit

## What I wanted

I wanted AI to review my proposed portfolio and out-of-sample backtest design before I started coding. I also wanted to check that Part B correctly continued the data foundation from my Part A project.

## Prompt

I asked AI to read the project brief, project context, data guide and AGENTS.md. I asked it to review equity-only, cryptocurrency-only and combined funds using equal weighting, minimum variance and risk parity. The proposed method used monthly walk-forward rebalancing, past data only, 252 observations for equity and combined funds, 365 observations for cryptocurrency funds, long-only weights, a 25% maximum weight, a zero risk-free rate and a 10-basis-point transaction cost.

I asked for a checklist against the brief, methodology risks, recommended tests, report assumptions and checks connecting Part B to Part A. I instructed it not to edit files or invent results.

## What the AI produced

The AI concluded that the proposed fund design met the higher-band portfolio requirements because it included three asset families and three portfolio methods. It identified possible look-ahead bias, calendar errors, unstable covariance estimates, optimisation problems and incorrect transaction-cost calculations.

It recommended tests for native-calendar returns, equity-calendar alignment, estimation windows, portfolio constraints, transaction costs, performance metrics, optimiser behaviour, sentiment lagging and required output files.

## What was useful

The AI correctly identified that the training data must end before each live rebalance date. It also confirmed that cryptocurrency returns must be calculated on the seven-day cryptocurrency calendar before being aligned with equity dates for the combined funds.

Its warning about unstable covariance estimates was useful because the combined fund has many assets compared with its 252-observation estimation window. I will therefore check that the optimisation succeeds and that different methods produce different weights.

## What was wrong or incomplete

The AI suggested running Python through `./.venv/bin/python`, but this project uses the already active environment shown in the PyCharm Terminal. I will use `python` instead.

It described sentiment, fusion and the Streamlit app as failed or missing parts of the portfolio design. They are required for the full Part B submission, but they were outside this first portfolio-design review and will be built in later steps.

The response did not clearly define annualised return. I will calculate annualised return geometrically from compounded daily fund returns rather than simply multiplying the average daily return.

## What I decided and why

I kept the proposed three asset families and three portfolio methods because they exceed the required minimum and allow a useful comparison.

I will use monthly walk-forward rebalancing and ensure that each live weight uses only earlier observations. Portfolio weights will be long-only, fully invested and capped at 25%.

One-way turnover will be calculated as half the sum of the absolute difference between the target weights and the drifted pre-trade weights. A cost of 0.10% will be applied to this turnover on rebalance dates.

I did not accept or paste any code from this review. The response was used to identify risks and decide which tests must be passed before accepting the portfolio implementation.
