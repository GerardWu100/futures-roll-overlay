"""Naive baselines for forward realized-variance forecasting."""

from __future__ import annotations

import pandas as pd


def trailing_variance_persistence(trailing_variance: pd.Series) -> pd.Series:
    """Use the latest observable trailing variance as the forward forecast.

    Parameters
    ----------
    trailing_variance : pd.Series
        Date-indexed annualized variance over the current and previous
        ``H - 1`` sessions. Every input is known at the forecast origin.

    Returns
    -------
    pd.Series
        Feasible persistence prediction aligned to the same index.
    """
    prediction = trailing_variance.copy()
    prediction.name = "prediction"
    return prediction
