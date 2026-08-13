"""BridgeSignal Funds — precomputed Part B investor app."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import streamlit as st


ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "data"
TABLE_DIR = ROOT / "results" / "tables"


st.set_page_config(
    page_title="BridgeSignal Funds",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --navy: #16324F;
        --teal: #2A9D8F;
        --coral: #E76F51;
    }

    .stApp {
        background: linear-gradient(
            180deg,
            #F8FAFC 0%,
            #FFFFFF 30%
        );
    }

    h1, h2, h3 {
        color: var(--navy);
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE5EC;
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 4px 14px rgba(22, 50, 79, 0.06);
    }

    .hero {
        background: linear-gradient(120deg, #16324F, #245A73);
        color: white;
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 18px;
    }

    .hero h1 {
        color: white;
        margin: 0 0 6px 0;
    }

    .hero p {
        color: #D7EEF2;
        margin: 0;
    }

    .note {
        background: #EFF8F6;
        border-left: 4px solid #2A9D8F;
        padding: 12px 14px;
        border-radius: 8px;
    }
    </style>

    <div class="hero">
        <h1>BridgeSignal Funds</h1>
        <p>
            Transparent multi-asset funds, risk evidence and
            lagged news analytics.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_results():
    """Load the precomputed Part B results."""

    paths = {
        "returns": DATA_DIR / "fund_returns.csv",
        "weights": DATA_DIR / "fund_weights.csv",
        "sentiment": DATA_DIR / "sector_sentiment_index.csv",
        "metrics": TABLE_DIR / "performance_metrics.csv",
    }

    missing = [
        str(path.relative_to(ROOT))
        for path in paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Run `python scripts/run_part_b.py` before opening the app. "
            "Missing: " + ", ".join(missing)
        )

    returns = pd.read_csv(
        paths["returns"],
        parse_dates=["date"],
    )
    weights = pd.read_csv(
        paths["weights"],
        parse_dates=["date"],
    )
    sentiment = pd.read_csv(
        paths["sentiment"],
        parse_dates=["date"],
    )
    metrics = pd.read_csv(paths["metrics"])

    return returns, weights, sentiment, metrics


def allocation_metrics(
    values: pd.Series,
    periods_per_year: int,
) -> dict:
    """Calculate performance measures for a custom allocation."""

    growth = (1 + values).cumprod()

    annual_return = (
        growth.iloc[-1] ** (periods_per_year / len(values)) - 1
    )

    volatility = (
        values.std(ddof=1) * np.sqrt(periods_per_year)
    )

    if volatility > 0:
        sharpe = (
            values.mean() * periods_per_year / volatility
        )
    else:
        sharpe = np.nan

    drawdown = growth / growth.cummax() - 1

    return {
        "Annual return": annual_return,
        "Annual volatility": volatility,
        "Sharpe": sharpe,
        "Maximum drawdown": drawdown.min(),
    }


try:
    (
        fund_returns,
        fund_weights,
        sector_sentiment,
        performance,
    ) = load_results()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


with st.sidebar:
    st.subheader("How to use the app")

    st.write(
        "1. Compare funds\n\n"
        "2. Open a fact sheet\n\n"
        "3. Build an allocation\n\n"
        "4. Explore sentiment"
    )

    st.divider()

    st.caption(
        "Historical out-of-sample results from the approximate "
        "2021–2023 live period. Not financial advice."
    )


(
    tab_compare,
    tab_fact,
    tab_allocate,
    tab_sentiment,
    tab_method,
) = st.tabs(
    [
        "Compare funds",
        "Fact sheet",
        "Allocation lab",
        "Sentiment lens",
        "Methodology",
    ]
)


with tab_compare:
    st.subheader("Compare the systematic fund range")

    family = st.segmented_control(
        "Fund family",
        options=["All", "Equity", "Crypto", "Combined"],
        default="All",
    )

    if family == "All":
        comparison = performance.copy()
    else:
        comparison = performance.loc[
            performance["family"] == family
        ].copy()

    view = comparison[
        [
            "fund",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "average_turnover",
        ]
    ].copy()

    percentage_columns = [
        "annual_return",
        "annual_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]

    for column in percentage_columns:
        view[column] = view[column].map(
            lambda value: f"{value:.2%}"
        )

    view["sharpe_ratio"] = view["sharpe_ratio"].map(
        lambda value: f"{value:.2f}"
    )

    view.columns = [
        "Fund",
        "Annual return",
        "Annual volatility",
        "Sharpe",
        "Maximum drawdown",
        "Average turnover",
    ]

    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
    )

    default_funds = (
        comparison
        .sort_values("sharpe_ratio", ascending=False)["fund"]
        .head(4)
        .tolist()
    )

    selected = st.multiselect(
        "Growth chart",
        comparison["fund"].tolist(),
        default=default_funds,
    )

    if selected:
        chart = (
            fund_returns.loc[
                fund_returns["fund"].isin(selected)
            ]
            .pivot(
                index="date",
                columns="fund",
                values="growth_of_1",
            )
        )

        st.line_chart(
            chart,
            y_label="Value of $1",
            x_label="Live backtest date",
        )
    else:
        st.info("Select at least one fund to display the chart.")


with tab_fact:
    st.subheader("Open a fund fact sheet")

    chosen = st.selectbox(
        "Fund",
        performance
        .sort_values(["family", "method"])["fund"],
    )

    row = performance.loc[
        performance["fund"] == chosen
    ].iloc[0]

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Annual return",
        f"{row['annual_return']:.2%}",
    )

    metric_columns[1].metric(
        "Annual volatility",
        f"{row['annual_volatility']:.2%}",
    )

    metric_columns[2].metric(
        "Sharpe",
        f"{row['sharpe_ratio']:.2f}",
    )

    metric_columns[3].metric(
        "Maximum drawdown",
        f"{row['maximum_drawdown']:.2%}",
    )

    st.caption(
        f"Live period: {row['first_live_date']} to "
        f"{row['last_live_date']} · "
        f"{int(row['estimation_window'])}-observation window · "
        f"monthly rebalancing · "
        f"{row['cost_bps']:.0f} bps turnover cost"
    )

    path = (
        fund_returns.loc[
            fund_returns["fund"] == chosen
        ]
        .set_index("date")
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Growth of $1**")

        st.line_chart(
            path[["growth_of_1"]],
            y_label="Value of $1",
        )

    with right:
        st.markdown("**Drawdown**")

        st.line_chart(
            100 * path[["drawdown"]],
            y_label="Drawdown (%)",
        )

    chosen_weights = fund_weights.loc[
        fund_weights["fund"] == chosen
    ]

    latest_date = chosen_weights["date"].max()

    holdings = (
        chosen_weights.loc[
            chosen_weights["date"] == latest_date,
            ["ticker", "weight"],
        ]
        .sort_values("weight", ascending=False)
        .assign(
            weight=lambda frame: frame["weight"].map(
                lambda value: f"{value:.2%}"
            )
        )
    )

    st.markdown(
        f"**Current holdings — latest rebalance "
        f"{latest_date:%d %b %Y}**"
    )

    st.dataframe(
        holdings,
        hide_index=True,
        width="stretch",
    )


with tab_allocate:
    st.subheader("Build an allocation across funds")

    st.write(
        "Select up to four funds. The chosen percentages are "
        "automatically adjusted to total 100%."
    )

    options = (
        performance
        .sort_values("fund")["fund"]
        .tolist()
    )

    defaults = [
        "Equity Minimum Variance",
        "Crypto Risk Parity",
        "Combined Risk Parity",
    ]

    selected = st.multiselect(
        "Funds",
        options,
        default=[
            name for name in defaults
            if name in options
        ],
        max_selections=4,
    )

    if not selected:
        st.info("Choose at least one fund.")

    else:
        input_columns = st.columns(len(selected))
        raw_allocations = {}

        for column, fund in zip(
            input_columns,
            selected,
        ):
            raw_allocations[fund] = column.number_input(
                f"{fund} (%)",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / len(selected),
                step=5.0,
            )

        total = sum(raw_allocations.values())

        if total <= 0:
            st.warning(
                "The total allocation must be greater than zero."
            )

        else:
            normalised = pd.Series(
                {
                    name: value / total
                    for name, value
                    in raw_allocations.items()
                }
            )

            return_matrix = (
                fund_returns.loc[
                    fund_returns["fund"].isin(selected)
                ]
                .pivot(
                    index="date",
                    columns="fund",
                    values="return",
                )[selected]
                .dropna()
            )

            blended_returns = return_matrix.mul(
                normalised,
                axis=1,
            ).sum(axis=1)

            selected_families = (
                performance
                .set_index("fund")
                .loc[selected, "family"]
            )

            if all(selected_families == "Crypto"):
                periods_per_year = 365
            else:
                periods_per_year = 252

            statistics = allocation_metrics(
                blended_returns,
                periods_per_year,
            )

            st.caption(
                "Normalised allocation: "
                + " · ".join(
                    f"{name}: {weight:.1%}"
                    for name, weight
                    in normalised.items()
                )
            )

            result_columns = st.columns(4)

            for column, (
                label,
                value,
            ) in zip(
                result_columns,
                statistics.items(),
            ):
                if label == "Sharpe":
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.2%}"

                column.metric(
                    label,
                    formatted_value,
                )

            custom_growth = (
                (1 + blended_returns)
                .cumprod()
                .rename("Custom allocation")
            )

            st.line_chart(
                custom_growth,
                y_label="Value of $1",
                x_label="Common live dates",
            )

            st.caption(
                "When fund calendars differ, the custom allocation "
                "uses dates available for every selected fund."
            )


with tab_sentiment:
    st.subheader("Lagged equity-sector news sentiment")

    sector = st.selectbox(
        "Sector",
        sorted(
            sector_sentiment["sector"].unique()
        ),
    )

    selected_sector = (
        sector_sentiment.loc[
            sector_sentiment["sector"] == sector
        ]
        .set_index("date")
    )

    sentiment_columns = st.columns(3)

    sentiment_columns[0].metric(
        "Average lagged sentiment",
        f"{selected_sector['tradable_sentiment_index'].mean():.3f}",
    )

    sentiment_columns[1].metric(
        "Average daily headlines",
        f"{selected_sector['article_count'].mean():.1f}",
    )

    sentiment_columns[2].metric(
        "Average coverage confidence",
        f"{selected_sector['coverage_confidence'].mean():.2f}",
    )

    st.line_chart(
        selected_sector[["smoothed_tradable_index"]],
        y_label="Lagged VADER score",
    )

    monthly_articles = (
        selected_sector[["article_count"]]
        .resample("ME")
        .sum()
    )

    st.bar_chart(
        monthly_articles,
        y_label="Headlines per month",
    )

    st.markdown(
        """
        <div class="note">
        Tickers are equally weighted, days without news are
        treated as neutral, and the signal is delayed by one
        equity trading day. The index measures headline tone,
        not whether the news is true or whether returns can
        be predicted.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_method:
    st.subheader("Transparent methodology")

    st.markdown(
        """
        - **Funds:** Equity-only, cryptocurrency-only and combined
          funds using Equal Weight, Minimum Variance and Risk Parity.
        - **Out-of-sample testing:** Each live weight uses only the
          earlier 252 equity or combined observations, or 365
          cryptocurrency observations. Funds are rebalanced monthly.
        - **Risk controls:** Portfolios are long-only, fully invested
          and limited to 25% per asset. Minimum Variance uses 20%
          covariance shrinkage.
        - **Costs:** Net returns deduct 10 basis points per unit of
          one-way turnover. The assumed risk-free rate is zero.
        - **Sentiment:** Finance-extended VADER preserves headline
          casing and punctuation. Tickers are equally weighted within
          sectors, and sentiment is delayed by one equity trading day.
        - **Innovation:** The Coverage-Aware Sentiment Tilt (CAST)
          combines the lagged 21-day ticker sentiment signal with the
          previous sector Coverage Confidence Index before adjusting
          Equity Minimum Variance weights.
        """
    )

    st.warning(
        "These backtests are historical evidence. They are not "
        "guarantees or personal investment recommendations."
    )
