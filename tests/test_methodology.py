"""Methodology tests for the Part B portfolio foundation."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parent.parent),
)

from src.features import (
    align_headlines_to_trading_days,
    combined_returns_panel,
    daily_returns,
)

from src.portfolios import (
    oos_backtest,
    optimise_weights,
    performance_metrics,
)

from src.sentiment import (
    sector_sentiment_index,
    ticker_daily_sentiment,
)


class MethodologyTests(unittest.TestCase):
    """Check the rules that protect the Part B results."""

    def test_returns_are_calculated_within_each_ticker(self):
        prices = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB", "BBB"],
                "date": pd.to_datetime(
                    [
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-01",
                        "2023-01-03",
                    ]
                ),
                "adjClose": [100.0, 110.0, 50.0, 55.0],
            }
        )

        result = daily_returns(prices)

        self.assertEqual(len(result), 2)
        self.assertTrue(
            np.allclose(result["return"], [0.10, 0.10])
        )

    def test_combined_panel_uses_equity_dates_only(self):
        equity = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "date": pd.to_datetime(
                    ["2023-01-06", "2023-01-09"]
                ),
                "return": [0.01, 0.02],
            }
        )

        crypto = pd.DataFrame(
            {
                "ticker": ["BTC", "BTC", "BTC", "BTC"],
                "date": pd.to_datetime(
                    [
                        "2023-01-06",
                        "2023-01-07",
                        "2023-01-08",
                        "2023-01-09",
                    ]
                ),
                "return": [0.03, 0.04, -0.02, 0.01],
            }
        )

        result = combined_returns_panel(equity, crypto)

        expected_dates = pd.DatetimeIndex(
            ["2023-01-06", "2023-01-09"],
            name="date",
        )

        pd.testing.assert_index_equal(
            result.index,
            expected_dates,
        )

        self.assertEqual(
            list(result.columns),
            ["EQ_AAA", "CR_BTC"],
        )

    def test_all_methods_produce_valid_weights(self):
        rng = np.random.default_rng(42)

        history = pd.DataFrame(
            rng.normal(0.0003, 0.01, size=(80, 5)),
            columns=["A", "B", "C", "D", "E"],
        )

        methods = [
            "equal_weight",
            "min_variance",
            "risk_parity",
        ]

        for method in methods:
            with self.subTest(method=method):
                weights = optimise_weights(
                    history,
                    method=method,
                    max_weight=0.25,
                )

                self.assertAlmostEqual(
                    float(weights.sum()),
                    1.0,
                )
                self.assertGreaterEqual(
                    float(weights.min()),
                    0.0,
                )
                self.assertLessEqual(
                    float(weights.max()),
                    0.25 + 1e-8,
                )

    def test_first_rebalance_does_not_use_future_returns(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range(
            "2020-01-01",
            periods=120,
        )

        base = pd.DataFrame(
            rng.normal(0.0003, 0.01, size=(120, 5)),
            index=dates,
            columns=["A", "B", "C", "D", "E"],
        )

        _, original_weights, original_metrics = oos_backtest(
            base,
            method="min_variance",
            family="Test",
            estimation_window=40,
            max_weight=0.25,
        )

        first_live_date = pd.Timestamp(
            original_metrics["first_live_date"]
        )

        changed = base.copy()
        changed.loc[first_live_date:, "A"] = 0.50
        changed.loc[first_live_date:, "B"] = -0.40

        _, changed_weights, _ = oos_backtest(
            changed,
            method="min_variance",
            family="Test",
            estimation_window=40,
            max_weight=0.25,
        )

        first_original = (
            original_weights.loc[
                original_weights["date"] == first_live_date
            ]
            .sort_values("ticker")["weight"]
            .to_numpy()
        )

        first_changed = (
            changed_weights.loc[
                changed_weights["date"] == first_live_date
            ]
            .sort_values("ticker")["weight"]
            .to_numpy()
        )

        np.testing.assert_allclose(
            first_original,
            first_changed,
            atol=1e-10,
        )

    def test_annualisation_uses_correct_calendar(self):
        returns = pd.Series(
            [0.01, -0.005, 0.003, 0.002, -0.001]
        )

        equity = performance_metrics(
            returns,
            periods_per_year=252,
        )

        crypto = performance_metrics(
            returns,
            periods_per_year=365,
        )

        expected_ratio = np.sqrt(365 / 252)

        actual_ratio = (
            crypto["annual_volatility"]
            / equity["annual_volatility"]
        )

        self.assertAlmostEqual(
            actual_ratio,
            expected_ratio,
        )

    def test_maximum_drawdown_uses_growth_path(self):
        metrics = performance_metrics(
            pd.Series([0.10, -0.20, 0.00]),
            periods_per_year=252,
        )

        self.assertAlmostEqual(
            metrics["maximum_drawdown"],
            -0.20,
        )

    def test_no_news_ticker_day_is_neutral_zero(self):
        dates = pd.DatetimeIndex(
            ["2023-01-09", "2023-01-10"]
        )

        universe = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "sector": ["Tech", "Tech"],
            }
        )

        scores = pd.DataFrame(
            {
                "trading_date": [
                    pd.Timestamp("2023-01-09")
                ],
                "ticker": ["AAA"],
                "sector": ["Tech"],
                "headline_sentiment": [0.8],
                "positive_share": [0.8],
                "neutral_share": [0.2],
                "negative_share": [0.0],
            }
        )

        result = ticker_daily_sentiment(
            scores,
            dates,
            universe,
        )

        no_news = result.loc[
            (
                result["trading_date"]
                == pd.Timestamp("2023-01-09")
            )
            & (result["ticker"] == "BBB")
        ].iloc[0]

        self.assertEqual(
            no_news["article_count"],
            0,
        )
        self.assertEqual(
            no_news["daily_sentiment"],
            0.0,
        )
        self.assertFalse(
            bool(no_news["has_news"])
        )

    def test_weekend_headline_is_first_tradable_on_tuesday(self):
        dates = pd.DatetimeIndex(
            [
                "2023-01-06",
                "2023-01-09",
                "2023-01-10",
            ]
        )

        headlines = pd.DataFrame(
            {
                "date": [
                    pd.Timestamp(
                        "2023-01-07",
                        tz="UTC",
                    )
                ],
                "ticker": ["AAA"],
                "sector": ["Tech"],
                "title": ["AAA beats forecasts"],
            }
        )

        aligned = align_headlines_to_trading_days(
            headlines,
            dates,
        )

        aligned["headline_sentiment"] = 0.8
        aligned["positive_share"] = 0.8
        aligned["neutral_share"] = 0.2
        aligned["negative_share"] = 0.0

        universe = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "sector": ["Tech"],
            }
        )

        result = ticker_daily_sentiment(
            aligned,
            dates,
            universe,
        )

        monday = result.loc[
            result["trading_date"]
            == pd.Timestamp("2023-01-09")
        ].iloc[0]

        tuesday = result.loc[
            result["trading_date"]
            == pd.Timestamp("2023-01-10")
        ].iloc[0]

        self.assertEqual(
            aligned.loc[0, "trading_date"],
            pd.Timestamp("2023-01-09"),
        )
        self.assertEqual(
            monday["lagged_daily_sentiment"],
            0.0,
        )
        self.assertEqual(
            tuesday["lagged_daily_sentiment"],
            0.8,
        )

    def test_sector_index_equal_weights_tickers(self):
        ticker_days = pd.DataFrame(
            {
                "trading_date": pd.to_datetime(
                    [
                        "2023-01-10",
                        "2023-01-10",
                    ]
                ),
                "ticker": ["AAA", "BBB"],
                "sector": ["Tech", "Tech"],
                "daily_sentiment": [1.0, -1.0],
                "lagged_daily_sentiment": [
                    1.0,
                    -1.0,
                ],
                "signal_21d": [1.0, -1.0],
                "article_count": [9, 1],
                "has_news": [True, True],
                "coverage_confidence": [
                    0.5,
                    0.5,
                ],
            }
        )

        result = sector_sentiment_index(
            ticker_days
        )

        self.assertAlmostEqual(
            result.loc[0, "sentiment_index"],
            0.0,
        )
        self.assertEqual(
            result.loc[0, "article_count"],
            10,
        )
if __name__ == "__main__":
    unittest.main(verbosity=2)
