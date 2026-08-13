"""Systematic funds and look-ahead-safe walk-forward backtests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize


METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "risk_parity": "Risk Parity",
}


def clean_return_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare a finite and date-sorted return matrix."""
    matrix = returns.copy()

    matrix.index = pd.DatetimeIndex(
        pd.to_datetime(matrix.index),
        name="date",
    )

    matrix = (
        matrix.sort_index()
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )

    if matrix.empty or matrix.shape[1] < 2:
        raise ValueError(
            "A complete return matrix with at least two assets is required."
        )

    return matrix


def shrink_covariance(
    history: pd.DataFrame,
    shrinkage: float = 0.20,
) -> np.ndarray:
    """Shrink the sample covariance towards its diagonal."""
    sample_covariance = history.cov().to_numpy(
        dtype=float
    )

    diagonal_covariance = np.diag(
        np.diag(sample_covariance)
    )

    covariance = (
        (1.0 - shrinkage) * sample_covariance
        + shrinkage * diagonal_covariance
    )

    return (
        covariance
        + np.eye(covariance.shape[0]) * 1e-10
    )


def bounded_normalise(
    weights: np.ndarray,
    maximum: float,
) -> np.ndarray:
    """Make weights non-negative, capped and fully invested."""
    result = np.clip(
        np.asarray(weights, dtype=float),
        0.0,
        None,
    )

    if result.sum() <= 0:
        result = np.ones_like(result)

    result = result / result.sum()

    # Raise an infeasible cap only for very small test portfolios.
    maximum = max(
        float(maximum),
        1.0 / len(result),
    )

    for _ in range(100):
        above_cap = result > maximum + 1e-12

        if not above_cap.any():
            break

        excess = float(
            (result[above_cap] - maximum).sum()
        )

        result[above_cap] = maximum
        below_cap = ~above_cap
        capacity = maximum - result[below_cap]

        if capacity.sum() <= 0:
            break

        result[below_cap] += (
            excess * capacity / capacity.sum()
        )

    return result / result.sum()


def optimise_weights(
    history: pd.DataFrame,
    method: str,
    max_weight: float = 0.25,
) -> pd.Series:
    """Create portfolio weights using historical observations only."""
    history = history.dropna(how="any")

    if history.empty:
        raise ValueError(
            "The estimation window contains no complete observations."
        )

    assets = list(history.columns)
    asset_count = len(assets)

    cap = max(
        float(max_weight),
        1.0 / asset_count,
    )

    equal_weights = np.repeat(
        1.0 / asset_count,
        asset_count,
    )

    if method == "equal_weight":
        return pd.Series(
            equal_weights,
            index=assets,
            name="weight",
        )

    if method not in METHOD_LABELS:
        raise ValueError(
            f"Unknown portfolio method: {method}"
        )

    covariance = shrink_covariance(history)

    constraints = (
        {
            "type": "eq",
            "fun": lambda weights: float(
                weights.sum() - 1.0
            ),
        },
    )

    bounds = [
        (0.0, cap)
        for _ in range(asset_count)
    ]

    if method == "min_variance":
        # Scaling prevents the optimiser from treating the
        # small daily covariance values as already converged.
        scaled_covariance = covariance * 10_000.0

        def objective(weights: np.ndarray) -> float:
            return float(
                weights
                @ scaled_covariance
                @ weights
            )

    else:

        def objective(weights: np.ndarray) -> float:
            portfolio_variance = float(
                weights @ covariance @ weights
            )

            if portfolio_variance <= 0:
                return 1e6

            marginal_risk = covariance @ weights

            risk_contributions = (
                weights
                * marginal_risk
                / portfolio_variance
            )

            equal_risk_target = np.repeat(
                1.0 / asset_count,
                asset_count,
            )

            return float(
                np.square(
                    risk_contributions
                    - equal_risk_target
                ).sum()
            )

    solution = minimize(
        objective,
        equal_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if (
        not solution.success
        or not np.isfinite(solution.x).all()
    ):
        final_weights = equal_weights
    else:
        final_weights = bounded_normalise(
            solution.x,
            cap,
        )

    return pd.Series(
        final_weights,
        index=assets,
        name="weight",
    )


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Calculate the required fund fact-sheet statistics."""
    values = pd.Series(
        daily_returns,
        dtype=float,
    ).dropna()

    if values.empty:
        raise ValueError(
            "At least one fund return is required."
        )

    growth = (1.0 + values).cumprod()
    years = len(values) / periods_per_year

    annual_return = float(
        growth.iloc[-1] ** (1.0 / years) - 1.0
    )

    annual_volatility = float(
        values.std(ddof=1)
        * math.sqrt(periods_per_year)
    )

    annual_excess_return = float(
        values.mean() * periods_per_year
        - risk_free_rate
    )

    if annual_volatility > 0:
        sharpe_ratio = (
            annual_excess_return
            / annual_volatility
        )
    else:
        sharpe_ratio = np.nan

    drawdown = (
        growth
        .div(growth.cummax())
        .sub(1.0)
    )

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "maximum_drawdown": float(
            drawdown.min()
        ),
        "final_growth_of_1": float(
            growth.iloc[-1]
        ),
        "observations": int(len(values)),
        "periods_per_year": int(
            periods_per_year
        ),
        "risk_free_rate": float(
            risk_free_rate
        ),
    }


def monthly_live_dates(
    index: pd.DatetimeIndex,
    estimation_window: int,
) -> list[pd.Timestamp]:
    """Select the first available live date in each month."""
    eligible_dates = index[estimation_window:]

    if len(eligible_dates) == 0:
        return []

    eligible = pd.Series(
        eligible_dates,
        index=eligible_dates,
    )

    return list(
        eligible.groupby(
            eligible.index.to_period("M"),
            sort=True,
        )
        .head(1)
        .index
    )


def calculate_return_path(
    returns: pd.DataFrame,
    scheduled_weights: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    """Apply monthly target weights and allow holdings to drift."""
    schedule = scheduled_weights.copy()
    schedule["date"] = pd.to_datetime(
        schedule["date"]
    )

    targets: dict[
        pd.Timestamp,
        pd.Series,
    ] = {}

    for date, group in schedule.groupby(
        "date",
        sort=True,
    ):
        target = (
            group.set_index("ticker")["weight"]
            .reindex(returns.columns)
        )

        if target.isna().any():
            raise ValueError(
                f"Missing scheduled weight at {date:%Y-%m-%d}"
            )

        targets[pd.Timestamp(date)] = (
            target.astype(float)
        )

    first_live_date = min(targets)

    live_returns = returns.loc[
        returns.index >= first_live_date
    ]

    current_weights: pd.Series | None = None
    records: list[dict] = []

    for date, asset_returns in live_returns.iterrows():
        turnover = 0.0

        if date in targets:
            target_weights = targets[date]

            if current_weights is not None:
                turnover = float(
                    (
                        target_weights
                        - current_weights
                    )
                    .abs()
                    .sum()
                    / 2.0
                )

            current_weights = (
                target_weights.copy()
            )

        if current_weights is None:
            continue

        gross_return = float(
            current_weights @ asset_returns
        )

        transaction_cost = (
            turnover
            * cost_bps
            / 10_000.0
        )

        net_return = (
            gross_return
            - transaction_cost
        )

        records.append(
            {
                "date": date,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": (
                    transaction_cost
                ),
                "return": net_return,
            }
        )

        # Holdings drift naturally with asset performance
        # until the next monthly rebalance.
        gross_growth = 1.0 + gross_return

        if gross_growth <= 0:
            raise ValueError(
                "Portfolio value became non-positive."
            )

        current_weights = (
            current_weights
            .mul(1.0 + asset_returns)
            / gross_growth
        )

    if not records:
        raise ValueError(
            "No live return periods were produced."
        )

    path = pd.DataFrame.from_records(records)

    path["growth_of_1"] = (
        1.0 + path["return"]
    ).cumprod()

    path["drawdown"] = (
        path["growth_of_1"]
        .div(
            path["growth_of_1"].cummax()
        )
        .sub(1.0)
    )

    return path


def oos_backtest(
    returns: pd.DataFrame,
    method: str = "min_variance",
    family: str = "Combined",
    estimation_window: int = 252,
    periods_per_year: int = 252,
    rebalance_frequency: str = "monthly",
    max_weight: float = 0.25,
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run a monthly rolling-window out-of-sample backtest."""
    if rebalance_frequency != "monthly":
        raise ValueError(
            "This project uses monthly rebalancing."
        )

    matrix = clean_return_matrix(returns)

    live_dates = monthly_live_dates(
        matrix.index,
        estimation_window,
    )

    if not live_dates:
        raise ValueError(
            "The estimation window is longer than the return sample."
        )

    method_label = METHOD_LABELS[method]
    fund_name = f"{family} {method_label}"

    weight_records: list[pd.DataFrame] = []

    for live_date in live_dates:
        position = int(
            matrix.index.get_loc(live_date)
        )

        # The current live date is excluded.
        history = matrix.iloc[
            position - estimation_window:
            position
        ]

        weights = optimise_weights(
            history,
            method=method,
            max_weight=max_weight,
        )

        weight_frame = (
            weights.rename_axis("ticker")
            .reset_index()
        )

        weight_frame["date"] = live_date
        weight_frame["fund"] = fund_name
        weight_frame["family"] = family
        weight_frame["method"] = method_label

        weight_records.append(
            weight_frame[
                [
                    "date",
                    "fund",
                    "family",
                    "method",
                    "ticker",
                    "weight",
                ]
            ]
        )

    weights_long = pd.concat(
        weight_records,
        ignore_index=True,
    )

    path = calculate_return_path(
        matrix,
        weights_long,
        cost_bps=cost_bps,
    )

    path["fund"] = fund_name
    path["family"] = family
    path["method"] = method_label
    path["periods_per_year"] = (
        periods_per_year
    )
    path["estimation_window"] = (
        estimation_window
    )
    path["cost_bps"] = cost_bps

    metrics = performance_metrics(
        path["return"],
        periods_per_year=periods_per_year,
        risk_free_rate=0.0,
    )

    positive_turnover = path.loc[
        path["turnover"] > 0,
        "turnover",
    ]

    metrics.update(
        {
            "fund": fund_name,
            "family": family,
            "method": method_label,
            "first_live_date": (
                path["date"]
                .min()
                .date()
                .isoformat()
            ),
            "last_live_date": (
                path["date"]
                .max()
                .date()
                .isoformat()
            ),
            "estimation_window": (
                estimation_window
            ),
            "rebalance_frequency": (
                rebalance_frequency
            ),
            "cost_bps": float(cost_bps),
            "average_turnover": float(
                positive_turnover.mean()
            ),
        }
    )

    return path, weights_long, metrics


def backtest_from_weights(
    returns: pd.DataFrame,
    scheduled_weights: pd.DataFrame,
    fund: str,
    family: str,
    method: str,
    periods_per_year: int = 252,
    estimation_window: int = 252,
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate an externally created monthly weight schedule."""
    matrix = clean_return_matrix(returns)

    path = calculate_return_path(
        matrix,
        scheduled_weights,
        cost_bps=cost_bps,
    )

    path["fund"] = fund
    path["family"] = family
    path["method"] = method
    path["periods_per_year"] = (
        periods_per_year
    )
    path["estimation_window"] = (
        estimation_window
    )
    path["cost_bps"] = cost_bps

    metrics = performance_metrics(
        path["return"],
        periods_per_year=periods_per_year,
        risk_free_rate=0.0,
    )

    positive_turnover = path.loc[
        path["turnover"] > 0,
        "turnover",
    ]

    metrics.update(
        {
            "fund": fund,
            "family": family,
            "method": method,
            "first_live_date": (
                path["date"]
                .min()
                .date()
                .isoformat()
            ),
            "last_live_date": (
                path["date"]
                .max()
                .date()
                .isoformat()
            ),
            "estimation_window": (
                estimation_window
            ),
            "rebalance_frequency": "monthly",
            "cost_bps": float(cost_bps),
            "average_turnover": float(
                positive_turnover.mean()
            ),
        }
    )

    return path, metrics
