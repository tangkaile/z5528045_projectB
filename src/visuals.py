"""Figures for the Part B report and application."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#16324F"
TEAL = "#2A9D8F"
AMBER = "#E9C46A"
CORAL = "#E76F51"
BLUE = "#4C78A8"
GREY = "#667085"

PALETTE = [
    NAVY,
    TEAL,
    CORAL,
    AMBER,
    BLUE,
    "#7A5195",
    "#6A994E",
]


def _style() -> None:
    """Apply a consistent BridgeSignal design."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "text.color": NAVY,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "font.size": 10,
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.alpha": 0.8,
        }
    )


def _save(
    figure: plt.Figure,
    path: pathlib.Path,
    note: str,
) -> None:
    """Add a source note and save a report-quality figure."""
    figure.text(
        0.01,
        0.01,
        note,
        fontsize=7.5,
        color=GREY,
    )

    figure.tight_layout(
        rect=(0, 0.035, 1, 0.95)
    )

    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)


def plot_growth(
    fund_returns: pd.DataFrame,
    output: pathlib.Path,
) -> None:
    """Plot the growth of one dollar for each fund."""
    families = [
        "Equity",
        "Crypto",
        "Combined",
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    figure.suptitle(
        "Figure 1. Out-of-sample growth of $1",
        fontsize=17,
        fontweight="bold",
    )

    for axis, family in zip(axes, families):
        subset = fund_returns.loc[
            fund_returns["family"] == family
        ]

        for color, (fund, data) in zip(
            PALETTE,
            subset.groupby("fund", sort=True),
        ):
            axis.plot(
                pd.to_datetime(data["date"]),
                data["growth_of_1"],
                label=fund.replace(
                    f"{family} ",
                    "",
                ),
                color=color,
                linewidth=1.8,
            )

        axis.set_title(family)
        axis.set_xlabel("Live backtest date")
        axis.set_ylabel("Value of $1")
        axis.legend(
            fontsize=8,
            frameon=False,
        )

    _save(
        figure,
        output,
        (
            "Net of 10 bps one-way turnover costs; "
            "risk-free rate = 0. Source: FINS3645 "
            "hosted data and the author's Python pipeline."
        ),
    )


def plot_drawdowns(
    fund_returns: pd.DataFrame,
    output: pathlib.Path,
) -> None:
    """Plot the drawdowns of the combined funds."""
    subset = fund_returns.loc[
        fund_returns["family"] == "Combined"
    ]

    figure, axis = plt.subplots(
        figsize=(11, 5.5)
    )

    figure.suptitle(
        "Figure 2. Combined-fund out-of-sample drawdowns",
        fontsize=17,
        fontweight="bold",
    )

    for color, (fund, data) in zip(
        PALETTE,
        subset.groupby("fund", sort=True),
    ):
        axis.plot(
            pd.to_datetime(data["date"]),
            100 * data["drawdown"],
            label=fund.replace(
                "Combined ",
                "",
            ),
            color=color,
            linewidth=1.8,
        )

    axis.axhline(
        0,
        color=NAVY,
        linewidth=0.8,
    )

    axis.set_xlabel("Live backtest date")
    axis.set_ylabel("Drawdown from prior peak (%)")
    axis.legend(
        frameon=False,
        ncol=2,
    )

    _save(
        figure,
        output,
        (
            "Drawdown is measured from each fund's "
            "previous high-water mark. Source: the "
            "author's Python pipeline."
        ),
    )


def _top_weight_panel(
    axis,
    data: pd.DataFrame,
    title: str,
) -> None:
    """Create one portfolio-weight panel."""
    pivot = (
        data.pivot(
            index="date",
            columns="ticker",
            values="weight",
        )
        .fillna(0)
    )

    top_tickers = (
        pivot.mean()
        .nlargest(7)
        .index
    )

    display = pivot[top_tickers].copy()

    display["Other"] = (
        1.0 - display.sum(axis=1)
    )

    axis.stackplot(
        pd.to_datetime(display.index),
        [
            display[column].to_numpy()
            for column in display.columns
        ],
        labels=list(display.columns),
        colors=PALETTE + ["#CBD5E1"],
        alpha=0.95,
    )

    axis.set_title(title)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Portfolio weight")

    axis.legend(
        loc="upper left",
        ncol=4,
        fontsize=7,
        frameon=False,
    )


def plot_weights(
    fund_weights: pd.DataFrame,
    output: pathlib.Path,
) -> None:
    """Compare combined-fund weights over time."""
    subset = fund_weights.loc[
        (
            fund_weights["family"]
            == "Combined"
        )
        & (
            fund_weights["method"].isin(
                [
                    "Minimum Variance",
                    "Risk Parity",
                ]
            )
        )
    ].copy()

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
    )

    figure.suptitle(
        "Figure 3. Combined-fund weights through time",
        fontsize=17,
        fontweight="bold",
    )

    methods = [
        "Minimum Variance",
        "Risk Parity",
    ]

    for axis, method in zip(axes, methods):
        _top_weight_panel(
            axis,
            subset.loc[
                subset["method"] == method
            ],
            method,
        )

    axes[-1].set_xlabel(
        "Monthly rebalance date"
    )

    _save(
        figure,
        output,
        (
            "Top seven assets are shown separately; "
            "remaining assets are grouped as Other. "
            "Long-only, fully invested and subject to "
            "a 25% asset cap."
        ),
    )


def plot_sharpe(
    metrics: pd.DataFrame,
    output: pathlib.Path,
) -> None:
    """Compare annualised Sharpe ratios."""
    ordered = metrics.sort_values(
        "sharpe_ratio"
    )

    colours = (
        ordered["family"]
        .map(
            {
                "Equity": TEAL,
                "Crypto": CORAL,
                "Combined": NAVY,
            }
        )
        .fillna(AMBER)
    )

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    figure.suptitle(
        "Figure 4. Out-of-sample Sharpe ratios",
        fontsize=17,
        fontweight="bold",
    )

    axis.barh(
        ordered["fund"],
        ordered["sharpe_ratio"],
        color=colours,
    )

    axis.axvline(
        0,
        color=NAVY,
        linewidth=0.8,
    )

    axis.set_xlabel(
        "Annualised Sharpe ratio"
    )

    axis.set_ylabel("Fund")

    _save(
        figure,
        output,
        (
            "Sharpe ratios use net returns, a zero "
            "risk-free rate and the appropriate 252-day "
            "or 365-day annualisation."
        ),
    )


def plot_sentiment(
    sentiment_index: pd.DataFrame,
    output: pathlib.Path,
) -> None:
    """Plot monthly sector sentiment as a heat map."""
    work = sentiment_index.copy()

    work["date"] = pd.to_datetime(
        work["date"]
    )

    monthly = (
        work.set_index("date")
        .groupby("sector")[
            "smoothed_tradable_index"
        ]
        .resample("ME")
        .mean()
        .reset_index()
    )

    pivot = monthly.pivot(
        index="sector",
        columns="date",
        values="smoothed_tradable_index",
    )

    limit = max(
        abs(float(np.nanmin(pivot.to_numpy()))),
        abs(float(np.nanmax(pivot.to_numpy()))),
        0.05,
    )

    figure, axis = plt.subplots(
        figsize=(14, 6)
    )

    figure.suptitle(
        "Figure 5. Lagged sector news-sentiment index",
        fontsize=17,
        fontweight="bold",
    )

    image = axis.imshow(
        pivot.to_numpy(),
        aspect="auto",
        cmap="RdYlGn",
        vmin=-limit,
        vmax=limit,
    )

    axis.set_yticks(
        np.arange(len(pivot.index)),
        labels=pivot.index,
    )

    positions = np.arange(
        0,
        len(pivot.columns),
        6,
    )

    axis.set_xticks(
        positions,
        labels=[
            pivot.columns[position].strftime(
                "%b\n%Y"
            )
            for position in positions
        ],
    )

    axis.set_xlabel("Trading month")
    axis.set_ylabel("Equity sector")

    colour_bar = figure.colorbar(
        image,
        ax=axis,
        pad=0.015,
    )

    colour_bar.set_label(
        "Smoothed, one-day-lagged VADER score"
    )

    _save(
        figure,
        output,
        (
            "Finance-extended VADER; tickers are "
            "equal-weighted within sectors; no-news "
            "ticker-days are neutral zero."
        ),
    )


def plot_fusion(
    fund_returns: pd.DataFrame,
    output: pathlib.Path,
) -> None:
    """Compare the base equity fund with CAST."""
    fund_names = [
        "Equity Minimum Variance",
        "Equity Coverage-Aware Sentiment",
    ]

    subset = fund_returns.loc[
        fund_returns["fund"].isin(
            fund_names
        )
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    figure.suptitle(
        "Figure 6. CAST before-versus-after comparison",
        fontsize=17,
        fontweight="bold",
    )

    for color, (fund, data) in zip(
        [NAVY, CORAL],
        subset.groupby(
            "fund",
            sort=False,
        ),
    ):
        dates = pd.to_datetime(
            data["date"]
        )

        axes[0].plot(
            dates,
            data["growth_of_1"],
            label=fund,
            color=color,
            linewidth=2,
        )

        axes[1].plot(
            dates,
            100 * data["drawdown"],
            label=fund,
            color=color,
            linewidth=2,
        )

    axes[0].set_title("Growth of $1")
    axes[0].set_ylabel("Value of $1")

    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown (%)")

    for axis in axes:
        axis.set_xlabel("Live backtest date")
        axis.legend(
            frameon=False,
            fontsize=8,
        )

    _save(
        figure,
        output,
        (
            "Before: Equity Minimum Variance. After: "
            "the same base weights tilted using lagged "
            "ticker sentiment and lagged sector CCI."
        ),
    )


def generate_all_figures(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    metrics: pd.DataFrame,
    sentiment_index: pd.DataFrame,
    output_directory: pathlib.Path,
) -> None:
    """Generate all six required report figures."""
    _style()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_growth(
        fund_returns,
        output_directory
        / "fund_growth_of_1.png",
    )

    plot_drawdowns(
        fund_returns,
        output_directory
        / "combined_drawdowns.png",
    )

    plot_weights(
        fund_weights,
        output_directory
        / "combined_weights_over_time.png",
    )

    plot_sharpe(
        metrics,
        output_directory
        / "fund_sharpe_ratios.png",
    )

    plot_sentiment(
        sentiment_index,
        output_directory
        / "sector_sentiment_index.png",
    )

    plot_fusion(
        fund_returns,
        output_directory
        / "fusion_before_after.png",
    )


__all__ = ["generate_all_figures"]
