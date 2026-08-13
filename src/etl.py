"""Station 1: load, clean and audit the three project datasets."""

from __future__ import annotations

import pandas as pd

from src import data_access


SAMPLE_END = pd.Timestamp("2023-12-31")
NEWS_DUPLICATE_KEY = ["ticker", "date", "title"]
PRICE_COLUMNS = ["open", "high", "low", "close", "adjClose", "volume"]


def normalise_date(series: pd.Series) -> pd.Series:
    """Convert dates to timezone-naive midnight timestamps."""
    converted = pd.to_datetime(series, utc=True, errors="coerce")
    return converted.dt.tz_convert(None).dt.normalize()


def clean_price_panel(
    prices: pd.DataFrame,
    sample_end: pd.Timestamp = SAMPLE_END,
) -> pd.DataFrame:
    """Clean an equity or crypto OHLCV panel."""
    clean = prices.copy()

    clean["date"] = normalise_date(clean["date"])
    clean = clean.loc[clean["date"] <= sample_end].copy()

    clean["ticker"] = clean["ticker"].astype("string").str.strip()

    numeric_columns = [
        column for column in PRICE_COLUMNS if column in clean.columns
    ]
    clean[numeric_columns] = clean[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # A return cannot be calculated without these fields.
    clean = clean.dropna(subset=["ticker", "date", "adjClose"])

    # Prices must be unique by ticker and date.
    clean = (
        clean.sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )

    return clean


def clean_news_headlines(
    headlines: pd.DataFrame,
    sample_end: pd.Timestamp = SAMPLE_END,
) -> pd.DataFrame:
    """Clean news without deleting valid same-day headlines."""
    clean = headlines.copy()

    clean["date"] = normalise_date(clean["date"])
    clean = clean.loc[clean["date"] <= sample_end].copy()

    text_columns = ["ticker", "sector", "title", "url", "publisher"]

    for column in text_columns:
        clean[column] = clean[column].astype("string").str.strip()

    clean = clean.dropna(
        subset=["ticker", "date", "sector", "title"]
    )
    clean = clean.loc[clean["title"] != ""]

    # Multiple different headlines for one ticker-day are valid.
    # Only an identical ticker-date-title combination is removed.
    clean = (
        clean.sort_values(NEWS_DUPLICATE_KEY)
        .drop_duplicates(NEWS_DUPLICATE_KEY, keep="first")
        .reset_index(drop=True)
    )

    return clean


def price_integrity_metrics(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    asset_class: str,
) -> dict:
    """Calculate the required price-data integrity checks."""
    raw_work = raw.copy()
    raw_work["date"] = normalise_date(raw_work["date"])
    raw_work = raw_work.loc[raw_work["date"] <= SAMPLE_END]

    duplicate_keys = int(
        raw_work.duplicated(["ticker", "date"]).sum()
    )

    missing_required = int(
        raw_work[["ticker", "date", "adjClose"]]
        .isna()
        .any(axis=1)
        .sum()
    )

    if asset_class.lower() == "crypto":
        expected_dates = pd.date_range(
            clean["date"].min(),
            clean["date"].max(),
            freq="D",
        )
    else:
        expected_dates = pd.DatetimeIndex(
            sorted(clean["date"].unique())
        )

    expected_index = pd.MultiIndex.from_product(
        [clean["ticker"].unique(), expected_dates],
        names=["ticker", "date"],
    )

    observed_index = pd.MultiIndex.from_frame(
        clean[["ticker", "date"]]
    )

    missing_ticker_dates = len(
        expected_index.difference(observed_index)
    )

    ohlc = clean[["open", "high", "low", "close"]]

    invalid_ohlc = (
        (clean["high"] < ohlc.max(axis=1))
        | (clean["low"] > ohlc.min(axis=1))
    )

    non_positive_prices = clean[
        ["open", "high", "low", "close", "adjClose"]
    ].le(0).any(axis=1)

    negative_volume = clean["volume"] < 0

    returns = (
        clean.sort_values(["ticker", "date"])
        .groupby("ticker")["adjClose"]
        .pct_change(fill_method=None)
    )

    extreme_returns = returns.abs() > 0.20

    return {
        "dataset": f"{asset_class.title()} prices",
        "raw_rows_to_2023": len(raw_work),
        "clean_rows": len(clean),
        "duplicate_ticker_date_keys": duplicate_keys,
        "missing_required_rows": missing_required,
        "missing_ticker_dates": int(missing_ticker_dates),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "non_positive_price_rows": int(
            non_positive_prices.sum()
        ),
        "negative_volume_rows": int(negative_volume.sum()),
        "absolute_return_over_20pct": int(
            extreme_returns.sum()
        ),
    }


def news_integrity_metrics(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    final_equity_date: pd.Timestamp,
) -> dict:
    """Calculate the required news-data integrity checks."""
    raw_work = raw.copy()
    raw_work["date"] = normalise_date(raw_work["date"])
    raw_work = raw_work.loc[raw_work["date"] <= SAMPLE_END]

    duplicate_keys = int(
        raw_work.duplicated(NEWS_DUPLICATE_KEY).sum()
    )

    missing_required = int(
        raw_work[["ticker", "date", "sector", "title"]]
        .isna()
        .any(axis=1)
        .sum()
    )

    blank_publishers = int(
        raw_work["publisher"].isna().sum()
        + raw_work["publisher"]
        .astype("string")
        .str.strip()
        .eq("")
        .sum()
    )

    unalignable_headlines = int(
        (clean["date"] > final_equity_date).sum()
    )

    return {
        "dataset": "News headlines",
        "raw_rows_to_2023": len(raw_work),
        "clean_rows": len(clean),
        "duplicate_ticker_date_title_keys": duplicate_keys,
        "missing_required_rows": missing_required,
        "blank_publisher_rows": blank_publishers,
        "headlines_after_final_equity_date": (
            unalignable_headlines
        ),
    }


def load_clean_equities() -> pd.DataFrame:
    """Load and clean the equity prices."""
    raw = data_access.load_equity_prices()
    return clean_price_panel(raw)


def load_clean_crypto() -> pd.DataFrame:
    """Load and clean crypto, including removal of 2024 rows."""
    raw = data_access.load_crypto_prices()
    return clean_price_panel(raw)


def load_clean_news() -> pd.DataFrame:
    """Load and exactly de-duplicate the news headlines."""
    raw = data_access.load_news_headlines()
    return clean_news_headlines(raw)
