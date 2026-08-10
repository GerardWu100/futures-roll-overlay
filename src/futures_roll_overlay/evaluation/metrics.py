"""Evaluation metrics for realized-variance forecast quality."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regression_metrics(
    actual: pd.Series,
    prediction: pd.Series,
) -> dict[str, float]:
    """Compute standard regression accuracy metrics.

    Parameters
    ----------
    actual : pd.Series
        Ground-truth target values.
    prediction : pd.Series
        Model prediction values aligned to ``actual`` index.

    Returns
    -------
    dict[str, float]
        Metric dictionary containing mean squared error, root mean squared
        error, mean absolute error, and ``R^2``.
    """
    aligned = pd.concat(
        [actual.rename("actual"), prediction.rename("prediction")], axis=1
    )
    aligned = aligned.dropna()
    if aligned.empty:
        # No overlapping observations means the metrics are undefined. Zeros
        # would read as a perfect fit, so report missing values instead.
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }

    residual = aligned["actual"] - aligned["prediction"]
    mse = float((residual.pow(2)).mean())
    rmse = float(np.sqrt(mse))
    mae = float(residual.abs().mean())

    total_variance = float(((aligned["actual"] - aligned["actual"].mean()) ** 2).sum())
    unexplained_variance = float((residual.pow(2)).sum())
    # Constant actuals make R^2 undefined; treat as zero explained variance.
    if total_variance == 0.0:
        r2 = 0.0
    else:
        r2 = 1.0 - (unexplained_variance / total_variance)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
    }
