"""Finance-aware VADER scoring and sector sentiment indices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import coverage_confidence_index


# Finance terms added to the standard VADER dictionary.
FINANCE_LEXICON = {
    "beat": 1.7,
    "beats": 1.7,
    "upgrade": 2.0,
    "upgrades": 2.0,
    "upgraded": 2.0,
    "outperform": 1.8,
    "bullish": 1.8,
    "surge": 1.8,
    "miss": -1.7,
    "misses": -1.7,
    "downgrade": -2.0,
    "downgrades": -2.0,
    "downgraded": -2.0,
    "underperform": -1.8,
    "bearish": -1.8,
    "plunge": -2.0,
    "bankruptcy": -3.0,
    "fraud": -2.7,
}


def _vader_analyser():
    """Load VADER and add the disclosed finance terms."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        analyser = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        analyser = SentimentIntensityAnalyzer()

    analyser.lexicon.update(FINANCE_LEXICON)
    return analyser


def score_headlines(
    aligned_headlines: pd.DataFrame,
    analyser=None,
) -> pd.DataFrame:
    """Score headlines while preserving their wording and punctuation."""
    if "title" not in aligned_headlines.columns:
        raise ValueError(
            "The aligned headline data needs a title column."
        )

    work = aligned_headlines.copy()
    analyser = analyser or _vader_analyser()

    scores = (
        work["title"]
        .fillna("")
        .astype(str)
        .map(analyser.polarity_scores)
    )

    score_frame = pd.DataFrame(
        scores.tolist(),
        index=work.index,
    ).rename(
        columns={
            "compound": "headline_sentiment",
            "pos": "positive_share",
            "neu": "neutral_share",
            "neg": "negative_share",
        }
    )

    return pd.concat(
        [work, score_frame],
        axis=1,
    )


def ticker_daily_sentiment(
    scores: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    universe: pd.DataFrame,
    signal_span: int = 21,
) -> pd.DataFrame:
    """Create daily ticker sentiment with neutral no-news days."""
    required = {
        "trading_date",
        "ticker",
        "sector",
        "headline_sentiment",
    }

    missing = required.difference(scores.columns)

    if missing:
        raise ValueError(
            f"Missing score columns: {sorted(missing)}"
        )

    grouped = (
        scores.groupby(
            ["trading_date", "ticker", "sector"],
            observed=True,
        )
        .agg(
            daily_sentiment=(
                "headline_sentiment",
                "mean",
            ),
            article_count=(
                "headline_sentiment",
                "size",
            ),
            positive_share=(
                "positive_share",
                "mean",
            ),
            neutral_share=(
                "neutral_share",
                "mean",
            ),
            negative_share=(
                "negative_share",
                "mean",
            ),
        )
        .reset_index()
    )

    dates = (
        pd.DatetimeIndex(pd.to_datetime(trading_dates))
        .sort_values()
        .unique()
    )

    names = (
        universe[["ticker", "sector"]]
        .drop_duplicates()
    )

    grid = (
        pd.MultiIndex.from_product(
            [dates, names["ticker"]],
            names=["trading_date", "ticker"],
        )
        .to_frame(index=False)
    )

    grid = grid.merge(
        names,
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    daily = grid.merge(
        grouped,
        on=["trading_date", "ticker", "sector"],
        how="left",
        validate="one_to_one",
    )

    daily["has_news"] = daily["article_count"].notna()

    daily["article_count"] = (
        daily["article_count"]
        .fillna(0)
        .astype(int)
    )

    score_columns = [
        "daily_sentiment",
        "positive_share",
        "neutral_share",
        "negative_share",
    ]

    for column in score_columns:
        daily[column] = daily[column].fillna(0.0)

    daily = (
        daily.sort_values(["ticker", "trading_date"])
        .reset_index(drop=True)
    )

    # A decision on day t can only use sentiment from day t-1.
    daily["lagged_daily_sentiment"] = (
        daily.groupby(
            "ticker",
            observed=True,
        )["daily_sentiment"]
        .shift(1)
        .fillna(0.0)
    )

    daily["signal_21d"] = (
        daily.groupby(
            "ticker",
            observed=True,
        )["lagged_daily_sentiment"]
        .transform(
            lambda values: values.ewm(
                span=signal_span,
                adjust=False,
            ).mean()
        )
    )

    coverage = coverage_confidence_index(daily)

    coverage = coverage.sort_values(
        ["sector", "trading_date"]
    )

    # Coverage is also lagged so the fusion remains look-ahead safe.
    coverage["lagged_coverage_confidence"] = (
        coverage.groupby(
            "sector",
            observed=True,
        )["coverage_confidence"]
        .shift(1)
        .fillna(0.0)
    )

    daily = daily.merge(
        coverage[
            [
                "trading_date",
                "sector",
                "coverage_confidence",
                "lagged_coverage_confidence",
            ]
        ],
        on=["trading_date", "sector"],
        how="left",
        validate="many_to_one",
    )

    daily["coverage_aware_signal"] = (
        daily["signal_21d"]
        * daily["lagged_coverage_confidence"]
    )

    return (
        daily.sort_values(
            ["trading_date", "sector", "ticker"]
        )
        .reset_index(drop=True)
    )


def sector_sentiment_index(
    ticker_days: pd.DataFrame,
) -> pd.DataFrame:
    """Equal-weight the five tickers within each equity sector."""
    required = {
        "trading_date",
        "ticker",
        "sector",
        "daily_sentiment",
        "lagged_daily_sentiment",
        "signal_21d",
        "article_count",
        "coverage_confidence",
        "has_news",
    }

    missing = required.difference(ticker_days.columns)

    if missing:
        raise ValueError(
            f"Missing ticker-day columns: {sorted(missing)}"
        )

    result = (
        ticker_days.groupby(
            ["trading_date", "sector"],
            observed=True,
        )
        .agg(
            sentiment_index=(
                "daily_sentiment",
                "mean",
            ),
            tradable_sentiment_index=(
                "lagged_daily_sentiment",
                "mean",
            ),
            smoothed_tradable_index=(
                "signal_21d",
                "mean",
            ),
            article_count=(
                "article_count",
                "sum",
            ),
            covered_tickers=(
                "has_news",
                "sum",
            ),
            coverage_confidence=(
                "coverage_confidence",
                "first",
            ),
        )
        .reset_index()
        .rename(
            columns={"trading_date": "date"}
        )
    )

    result["covered_tickers"] = (
        result["covered_tickers"].astype(int)
    )

    return (
        result.sort_values(["date", "sector"])
        .reset_index(drop=True)
    )


def sentiment_validation_table(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise positive, neutral and negative headline scores."""
    work = scores.copy()

    work["classification"] = np.select(
        [
            work["headline_sentiment"] >= 0.05,
            work["headline_sentiment"] <= -0.05,
        ],
        ["Positive", "Negative"],
        default="Neutral",
    )

    summary = (
        work.groupby(
            "classification",
            observed=True,
        )
        .agg(
            headlines=("title", "size"),
            mean_score=(
                "headline_sentiment",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["share"] = (
        summary["headlines"]
        / summary["headlines"].sum()
    )

    return summary


__all__ = [
    "FINANCE_LEXICON",
    "score_headlines",
    "ticker_daily_sentiment",
    "sector_sentiment_index",
    "sentiment_validation_table",
]
