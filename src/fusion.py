"""Coverage-Aware Sentiment Tilt for the equity fund."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cap_weights(
    weights: pd.Series,
    maximum: float,
) -> pd.Series:
    """Keep weights non-negative, capped and fully invested."""
    result = weights.clip(lower=0.0).astype(float)

    if result.sum() <= 0:
        result[:] = 1.0

    result = result / result.sum()

    maximum = max(
        float(maximum),
        1.0 / len(result),
    )

    for _ in range(100):
        over_limit = result > maximum + 1e-12

        if not over_limit.any():
            break

        excess = float(
            (
                result[over_limit]
                - maximum
            ).sum()
        )

        result.loc[over_limit] = maximum

        under_limit = ~over_limit
        capacity = (
            maximum
            - result.loc[under_limit]
        )

        if capacity.sum() <= 0:
            break

        result.loc[under_limit] += (
            excess
            * capacity
            / capacity.sum()
        )

    return result / result.sum()


def apply_sentiment(
    weights: pd.DataFrame,
    sentiment: pd.DataFrame,
    strength: float = 0.50,
    max_weight: float = 0.25,
    fund_name: str = (
        "Equity Coverage-Aware Sentiment"
    ),
) -> pd.DataFrame:
    """Tilt equity weights using lagged sentiment and coverage.

    The signal combines the 21-day lagged ticker sentiment with
    the previous trading day's sector Coverage Confidence Index.
    Positive signals increase a weight and negative signals reduce
    it, while the original portfolio remains the starting point.
    """
    required_weights = {
        "date",
        "ticker",
        "weight",
    }

    required_signal = {
        "trading_date",
        "ticker",
        "coverage_aware_signal",
        "lagged_coverage_confidence",
    }

    missing_weights = required_weights.difference(
        weights.columns
    )

    if missing_weights:
        raise ValueError(
            f"Missing weight columns: "
            f"{sorted(missing_weights)}"
        )

    missing_signal = required_signal.difference(
        sentiment.columns
    )

    if missing_signal:
        raise ValueError(
            f"Missing sentiment columns: "
            f"{sorted(missing_signal)}"
        )

    signals = sentiment[
        [
            "trading_date",
            "ticker",
            "coverage_aware_signal",
            "lagged_coverage_confidence",
        ]
    ].rename(
        columns={"trading_date": "date"}
    )

    work = weights.merge(
        signals,
        on=["date", "ticker"],
        how="left",
        validate="many_to_one",
    )

    signal_columns = [
        "coverage_aware_signal",
        "lagged_coverage_confidence",
    ]

    work[signal_columns] = (
        work[signal_columns].fillna(0.0)
    )

    frames = []

    for date, group in work.groupby(
        "date",
        sort=True,
    ):
        group = group.copy()

        signal = group[
            "coverage_aware_signal"
        ]

        standard_deviation = float(
            signal.std(ddof=0)
        )

        if standard_deviation > 1e-12:
            z_score = (
                signal - signal.mean()
            ) / standard_deviation
        else:
            z_score = pd.Series(
                0.0,
                index=group.index,
            )

        tilt_multiplier = np.exp(
            strength * z_score
        )

        raw_weights = (
            group["weight"]
            * tilt_multiplier
        )

        tilted_weights = _cap_weights(
            pd.Series(
                raw_weights.to_numpy(),
                index=group["ticker"],
            ),
            maximum=max_weight,
        )

        group["base_weight"] = group["weight"]

        group["weight"] = (
            group["ticker"]
            .map(tilted_weights)
        )

        group["signal_z_score"] = (
            z_score.to_numpy()
        )

        group["tilt_multiplier"] = (
            tilt_multiplier.to_numpy()
        )

        group["fund"] = fund_name
        group["family"] = "Equity"

        group["method"] = (
            "Coverage-Aware Sentiment"
        )

        frames.append(group)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return result[
        [
            "date",
            "fund",
            "family",
            "method",
            "ticker",
            "weight",
            "base_weight",
            "coverage_aware_signal",
            "lagged_coverage_confidence",
            "signal_z_score",
            "tilt_multiplier",
        ]
    ]


__all__ = ["apply_sentiment"]
