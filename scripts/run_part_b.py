"""Reproduce all Part B results.

Run from the project folder using:

    python scripts/run_part_b.py
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT),
)

from src.etl import (
    load_clean_crypto,
    load_clean_equities,
    load_clean_news,
)

from src.features import (
    align_headlines_to_trading_days,
    combined_returns_panel,
    daily_returns,
    wide_returns,
)

from src.fusion import apply_sentiment

from src.portfolios import (
    METHOD_LABELS,
    backtest_from_weights,
    oos_backtest,
)

from src.sentiment import (
    score_headlines,
    sector_sentiment_index,
    sentiment_validation_table,
    ticker_daily_sentiment,
)

from src.visuals import generate_all_figures


DATA_DIRECTORY = ROOT / "results" / "data"
TABLE_DIRECTORY = ROOT / "results" / "tables"
FIGURE_DIRECTORY = ROOT / "results" / "figures"

METHODS = [
    "equal_weight",
    "min_variance",
    "risk_parity",
]


def main() -> None:
    """Run the full Part B pipeline."""
    for directory in [
        DATA_DIRECTORY,
        TABLE_DIRECTORY,
        FIGURE_DIRECTORY,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "1/7 Loading and cleaning course data..."
    )

    equities = load_clean_equities()
    crypto = load_clean_crypto()
    news = load_clean_news()

    print("Equities:", equities.shape)
    print("Crypto:", crypto.shape)
    print("News:", news.shape)

    print(
        "2/7 Calculating native-calendar returns..."
    )

    equity_returns = daily_returns(
        equities,
        asset_class="Equity",
    )

    crypto_returns = daily_returns(
        crypto,
        asset_class="Crypto",
    )

    equity_matrix = (
        wide_returns(equity_returns)
        .dropna(how="any")
    )

    crypto_matrix = (
        wide_returns(crypto_returns)
        .dropna(how="any")
    )

    combined_matrix = (
        combined_returns_panel(
            equity_returns,
            crypto_returns,
        )
        .dropna(how="any")
    )

    print("Equity matrix:", equity_matrix.shape)
    print("Crypto matrix:", crypto_matrix.shape)
    print("Combined matrix:", combined_matrix.shape)

    print(
        "3/7 Running nine walk-forward backtests..."
    )

    family_inputs = {
        "Equity": (
            equity_matrix,
            252,
            252,
        ),
        "Crypto": (
            crypto_matrix,
            365,
            365,
        ),
        "Combined": (
            combined_matrix,
            252,
            252,
        ),
    }

    return_frames = []
    weight_frames = []
    metric_records = []

    for family, values in family_inputs.items():
        matrix, window, annualisation = values

        for method in METHODS:
            print(
                "   ",
                family,
                "-",
                METHOD_LABELS[method],
            )

            path, weights, metrics = oos_backtest(
                matrix,
                method=method,
                family=family,
                estimation_window=window,
                periods_per_year=annualisation,
                rebalance_frequency="monthly",
                max_weight=0.25,
                cost_bps=10.0,
            )

            return_frames.append(path)
            weight_frames.append(weights)
            metric_records.append(metrics)

    print(
        "4/7 Building the sector sentiment index..."
    )

    trading_dates = pd.DatetimeIndex(
        equities["date"]
        .sort_values()
        .unique()
    )

    aligned_news = (
        align_headlines_to_trading_days(
            news,
            trading_dates,
        )
    )

    scored_news = score_headlines(
        aligned_news
    )

    universe = (
        equities[["ticker", "sector"]]
        .drop_duplicates()
    )

    ticker_sentiment = ticker_daily_sentiment(
        scored_news,
        trading_dates,
        universe,
    )

    sector_index = sector_sentiment_index(
        ticker_sentiment
    )

    sentiment_validation = (
        sentiment_validation_table(
            scored_news
        )
    )

    print(
        "5/7 Applying the Coverage-Aware "
        "Sentiment Tilt..."
    )

    core_weights = pd.concat(
        weight_frames,
        ignore_index=True,
    )

    base_weights = core_weights.loc[
        core_weights["fund"]
        == "Equity Minimum Variance"
    ].copy()

    cast_weights = apply_sentiment(
        base_weights,
        ticker_sentiment,
        strength=0.50,
        max_weight=0.25,
    )

    cast_path, cast_metrics = (
        backtest_from_weights(
            equity_matrix,
            cast_weights,
            fund=(
                "Equity Coverage-Aware Sentiment"
            ),
            family="Equity",
            method=(
                "Coverage-Aware Sentiment"
            ),
            periods_per_year=252,
            estimation_window=252,
            cost_bps=10.0,
        )
    )

    return_frames.append(cast_path)
    weight_frames.append(cast_weights)
    metric_records.append(cast_metrics)

    print(
        "6/7 Saving tables, data and figures..."
    )

    fund_returns = pd.concat(
        return_frames,
        ignore_index=True,
    )

    fund_weights = pd.concat(
        weight_frames,
        ignore_index=True,
        sort=False,
    )

    performance_metrics = pd.DataFrame(
        metric_records
    )

    metric_columns = [
        "fund",
        "family",
        "method",
        "first_live_date",
        "last_live_date",
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "final_growth_of_1",
        "observations",
        "periods_per_year",
        "estimation_window",
        "rebalance_frequency",
        "risk_free_rate",
        "cost_bps",
        "average_turnover",
    ]

    performance_metrics = (
        performance_metrics[metric_columns]
        .sort_values(
            ["family", "method"]
        )
        .reset_index(drop=True)
    )

    # These four filenames are required by the brief.
    fund_returns.to_csv(
        DATA_DIRECTORY / "fund_returns.csv",
        index=False,
    )

    fund_weights.to_csv(
        DATA_DIRECTORY / "fund_weights.csv",
        index=False,
    )

    sector_index.to_csv(
        DATA_DIRECTORY
        / "sector_sentiment_index.csv",
        index=False,
    )

    performance_metrics.to_csv(
        TABLE_DIRECTORY
        / "performance_metrics.csv",
        index=False,
    )

    # Extra evidence files.
    ticker_sentiment.to_csv(
        DATA_DIRECTORY
        / "ticker_sentiment_signal.csv",
        index=False,
    )

    sentiment_validation.to_csv(
        TABLE_DIRECTORY
        / "sentiment_validation.csv",
        index=False,
    )

    latest_dates = (
        fund_weights.groupby(
            "fund",
            observed=True,
        )["date"]
        .transform("max")
    )

    current_holdings = fund_weights.loc[
        pd.to_datetime(fund_weights["date"])
        == pd.to_datetime(latest_dates)
    ].copy()

    current_holdings = (
        current_holdings.sort_values(
            ["fund", "weight"],
            ascending=[True, False],
        )
    )

    current_holdings.to_csv(
        TABLE_DIRECTORY
        / "current_holdings.csv",
        index=False,
    )

    fusion_comparison = (
        performance_metrics.loc[
            performance_metrics["fund"].isin(
                [
                    "Equity Minimum Variance",
                    (
                        "Equity Coverage-Aware "
                        "Sentiment"
                    ),
                ]
            )
        ]
    )

    fusion_comparison.to_csv(
        TABLE_DIRECTORY
        / "fusion_before_after.csv",
        index=False,
    )

    assumptions = pd.DataFrame(
        [
            [
                "Risk-free rate",
                "0%",
                "A simple disclosed assumption",
            ],
            [
                "Rebalancing",
                "Monthly",
                (
                    "First available live date "
                    "of each month"
                ),
            ],
            [
                "Equity and combined window",
                "252 observations",
                "About one equity trading year",
            ],
            [
                "Crypto window",
                "365 observations",
                "One crypto calendar year",
            ],
            [
                "Transaction cost",
                "10 bps per one-way turnover",
                "Applied on rebalance dates",
            ],
            [
                "No-news treatment",
                "Neutral zero",
                (
                    "Keeps a complete equal-weight "
                    "ticker panel"
                ),
            ],
            [
                "Sentiment lag",
                "One equity trading day",
                (
                    "Weekend and Monday news first "
                    "becomes usable on Tuesday"
                ),
            ],
            [
                "Portfolio constraints",
                "Long-only and 25% maximum",
                "Limits concentration",
            ],
        ],
        columns=[
            "assumption",
            "choice",
            "justification",
        ],
    )

    assumptions.to_csv(
        TABLE_DIRECTORY
        / "model_assumptions.csv",
        index=False,
    )

    generate_all_figures(
        fund_returns,
        fund_weights,
        performance_metrics,
        sector_index,
        FIGURE_DIRECTORY,
    )

    print("7/7 Pipeline complete.")
    print(
        "Funds created:",
        len(performance_metrics),
    )

    print(
        performance_metrics[
            [
                "fund",
                "annual_return",
                "annual_volatility",
                "sharpe_ratio",
                "maximum_drawdown",
            ]
        ].to_string(index=False)
    )

    print(
        "Required outputs saved under results/."
    )


if __name__ == "__main__":
    main()
