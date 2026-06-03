"""Tests for explainable model training functions."""

from __future__ import annotations

import pandas as pd

from futures_roll_overlay.models.train import fit_ridge_regression


def test_fit_ridge_regression_returns_coefficients_and_predictions() -> None:
    """Ridge trainer should return in-sample predictions and coefficient table."""
    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [0.1, 0.2, 0.3, 0.4],
            "target_forward_rv_annualized": [0.05, 0.06, 0.07, 0.08],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    result = fit_ridge_regression(
        dataset=frame,
        feature_columns=["feature_a", "feature_b"],
        target_column="target_forward_rv_annualized",
        alpha=1.0,
    )
    assert "prediction" in result.predictions.columns
    assert len(result.coefficients) == 2
