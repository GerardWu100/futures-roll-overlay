"""Naive baselines for forward realized-variance forecasting."""

from __future__ import annotations

import pandas as pd


def persistence_baseline(target: pd.Series) -> pd.Series:
    """Predict next value using the previous observed target.

    Parameters
    ----------
    target : pd.Series
        Date-indexed target series.

    Returns
    -------
    pd.Series
        One-step lagged prediction series aligned to the same index.
    """
    prediction = target.shift(1)
    prediction.name = "prediction"
    return prediction
