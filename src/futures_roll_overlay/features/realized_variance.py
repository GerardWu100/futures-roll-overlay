"""Realized-variance target construction from daily log returns."""

from __future__ import annotations

import pandas as pd


def build_forward_realized_variance(
    log_returns: pd.Series,
    horizon_days: int,
    annualization_factor: int,
) -> pd.DataFrame:
    """Build daily and forward-horizon realized-variance targets.

    Definitions used by this function:

    - ``r_t``: daily log return on date ``t``.
    - ``rv_t = r_t^2``: daily realized variance.
    - ``RV_{t,t+H} = sum_{i=1}^{H} r_{t+i}^2``: forward realized variance over
      ``H`` days.
    - ``RV_annualized = (annualization_factor / H) * RV_{t,t+H}``.

    Parameters
    ----------
    log_returns : pd.Series
        Date-indexed daily log-return series.
    horizon_days : int
        Forward horizon ``H`` in trading days.
    annualization_factor : int
        Trading days per year for annualization, typically ``252``.

    Returns
    -------
    pd.DataFrame
        Date-indexed frame with columns:

        - ``log_return``

        - ``daily_realized_variance``
        - ``known_trailing_rv_annualized``
        - ``forward_realized_variance``
        - ``forward_realized_variance_annualized``
    """
    log_returns = log_returns.copy()
    log_returns.index = pd.to_datetime(log_returns.index)
    log_returns.index.name = "date"
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")
    if annualization_factor < 1:
        raise ValueError("annualization_factor must be at least 1.")

    squared_returns = log_returns.pow(2)
    # Build the target from explicit leads. A shifted rolling window still
    # looks backward and would attach r_t^2 + ... + r_{t+H-1}^2 to date t.
    future_squared_returns = [
        squared_returns.shift(-lead) for lead in range(1, horizon_days + 1)
    ]
    forward_sum = pd.concat(future_squared_returns, axis=1).sum(
        axis=1,
        min_count=horizon_days,
    )
    trailing_sum = squared_returns.rolling(
        window=horizon_days,
        min_periods=horizon_days,
    ).sum()
    annualized_forward_sum = (annualization_factor / horizon_days) * forward_sum
    output = pd.DataFrame(
        {
            "log_return": log_returns,
            "daily_realized_variance": squared_returns,
            "known_trailing_rv_annualized": (annualization_factor / horizon_days)
            * trailing_sum,
            "forward_realized_variance": forward_sum,
            "forward_realized_variance_annualized": annualized_forward_sum,
        }
    )
    output.index.name = "date"
    return output
