"""Part A return and headline features reused in Part B."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
    asset_class: str | None = None,
) -> pd.DataFrame:
    """Calculate simple returns within each ticker's native calendar."""
    required_columns = {"ticker", "date", price_col}
    missing_columns = required_columns.difference(prices.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    columns = ["ticker", "date", price_col]

    if "sector" in prices.columns:
        columns.append("sector")

    result = (
        prices[columns]
        .copy()
        .sort_values(["ticker", "date"])
    )

    result["return"] = (
        result.groupby("ticker", observed=True)[price_col]
        .pct_change(fill_method=None)
    )

    # The first observation for each ticker has no previous price.
    result = result.dropna(subset=["return"])

    if asset_class is not None:
        result["asset_class"] = asset_class

    return result.reset_index(drop=True)


def wide_returns(
    returns: pd.DataFrame,
    prefix: str = "",
) -> pd.DataFrame:
    """Convert long ticker returns into a date-by-ticker matrix."""
    wide = (
        returns.pivot(
            index="date",
            columns="ticker",
            values="return",
        )
        .sort_index()
    )

    if prefix:
        wide.columns = [
            f"{prefix}{ticker}"
            for ticker in wide.columns
        ]

    wide.index = pd.DatetimeIndex(
        wide.index,
        name="date",
    )

    return wide


def combined_returns_panel(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Join calculated crypto returns onto the equity trading calendar."""
    equity_wide = wide_returns(
        equity_returns,
        prefix="EQ_",
    )

    crypto_wide = wide_returns(
        crypto_returns,
        prefix="CR_",
    )

    # Keeping equities on the left means the combined fund
    # operates only on equity trading dates.
    combined = equity_wide.join(
        crypto_wide,
        how="left",
    )

    return combined


def align_headlines_to_trading_days(
    headlines: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map headlines to the same or next equity trading day."""
    dates = (
        pd.DatetimeIndex(pd.to_datetime(trading_dates))
        .sort_values()
        .unique()
    )

    if dates.tz is not None:
        dates = dates.tz_convert(None)

    work = headlines.copy()

    work["date"] = (
        pd.to_datetime(work["date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    positions = dates.searchsorted(
        work["date"],
        side="left",
    )

    valid = positions < len(dates)
    work = work.loc[valid].copy()

    work["trading_date"] = (
        dates.to_numpy()[positions[valid]]
    )

    return (
        work.sort_values(
            ["trading_date", "ticker", "date"]
        )
        .reset_index(drop=True)
    )


def assemble_headline_panel(
    headlines: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a complete trading-day and ticker headline panel."""
    aligned = align_headlines_to_trading_days(
        headlines,
        trading_dates,
    )

    grouped = (
        aligned.groupby(
            ["trading_date", "ticker", "sector"],
            observed=True,
        )
        .agg(
            article_count=("title", "size"),
            headline_text=(
                "title",
                lambda values: " || ".join(
                    values.astype(str)
                ),
            ),
            unique_publisher_count=(
                "publisher",
                lambda values: (
                    values.dropna()
                    .replace("", pd.NA)
                    .nunique()
                ),
            ),
        )
        .reset_index()
    )

    dates = (
        pd.DatetimeIndex(pd.to_datetime(trading_dates))
        .sort_values()
        .unique()
    )

    universe = (
        headlines[["ticker", "sector"]]
        .drop_duplicates()
    )

    complete_grid = pd.MultiIndex.from_product(
        [dates, universe["ticker"]],
        names=["trading_date", "ticker"],
    ).to_frame(index=False)

    complete_grid = complete_grid.merge(
        universe,
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    panel = complete_grid.merge(
        grouped,
        on=["trading_date", "ticker", "sector"],
        how="left",
        validate="one_to_one",
    )

    for column in [
        "article_count",
        "unique_publisher_count",
    ]:
        panel[column] = (
            panel[column]
            .fillna(0)
            .astype(int)
        )

    panel["headline_text"] = (
        panel["headline_text"].fillna("")
    )

    return (
        panel.sort_values(
            ["trading_date", "sector", "ticker"]
        )
        .reset_index(drop=True)
    )


def coverage_confidence_index(
    ticker_days: pd.DataFrame,
    tickers_per_sector: int = 5,
    depth_scale: float = 5.0,
) -> pd.DataFrame:
    """Reproduce the Part A Coverage Confidence Index."""
    work = ticker_days[
        [
            "trading_date",
            "sector",
            "ticker",
            "article_count",
        ]
    ].copy()

    sector_totals = work.groupby(
        ["trading_date", "sector"],
        observed=True,
    )["article_count"].transform("sum")

    article_shares = work["article_count"].div(
        sector_totals.where(sector_totals > 0)
    )

    work["share_squared"] = (
        article_shares.pow(2).fillna(0)
    )

    daily = (
        work.groupby(
            ["trading_date", "sector"],
            observed=True,
        )
        .agg(
            article_count=("article_count", "sum"),
            covered_tickers=(
                "article_count",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
            concentration_hhi=(
                "share_squared",
                "sum",
            ),
        )
        .reset_index()
    )

    daily["effective_tickers"] = np.where(
        daily["concentration_hhi"] > 0,
        1.0 / daily["concentration_hhi"],
        0.0,
    )

    daily["depth_saturation"] = (
        1.0
        - np.exp(
            -daily["article_count"] / depth_scale
        )
    )

    daily["coverage_confidence"] = (
        daily["effective_tickers"]
        / tickers_per_sector
        * daily["depth_saturation"]
    ).clip(0, 1)

    return daily
