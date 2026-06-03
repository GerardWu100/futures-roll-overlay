"""Explainable model training utilities for realized-variance forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RidgeTrainingResult:
    """Container for ridge-model fit outputs.

    Attributes
    ----------
    predictions : pd.DataFrame
        DataFrame indexed like the input dataset with ``prediction`` and
        ``actual`` columns.
    coefficients : pd.DataFrame
        Coefficient table with columns ``feature`` and ``coefficient``.
    intercept : float
        Scalar intercept value for the fitted ridge model.
    """

    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    intercept: float


def fit_feature_standardizer(
    feature_frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Compute column means and standard deviations for z-scoring features.

    Zero-variance columns are mapped to unit scale so division stays stable.
    """
    means = feature_frame.mean(axis=0)
    stds = feature_frame.std(axis=0, ddof=0).replace(0.0, 1.0)
    return means, stds


def apply_feature_standardizer(
    feature_frame: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    """Apply training-set moments to a feature block (for example OOS test rows)."""
    return (feature_frame - means) / stds


def _zscore_frame(
    feature_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardize one feature block and return transformed data plus moments."""
    means, stds = fit_feature_standardizer(feature_frame=feature_frame)
    standardized = apply_feature_standardizer(
        feature_frame=feature_frame,
        means=means,
        stds=stds,
    )
    return standardized, means, stds


def _ridge_closed_form(
    design_matrix: np.ndarray,
    target_vector: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Solve ridge coefficients with closed-form normal equations."""
    feature_count = design_matrix.shape[1]
    identity = np.eye(feature_count)
    # Do not penalize the intercept column in the ridge penalty matrix.
    identity[0, 0] = 0.0
    left_hand_side = design_matrix.T @ design_matrix + alpha * identity
    right_hand_side = design_matrix.T @ target_vector
    return np.linalg.solve(left_hand_side, right_hand_side)


def fit_ridge_regression(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    alpha: float,
) -> RidgeTrainingResult:
    """Fit an explainable ridge regression using NumPy linear algebra.

    Parameters
    ----------
    dataset : pd.DataFrame
        Training panel containing all requested feature and target columns.
    feature_columns : list[str]
        Ordered feature column names.
    target_column : str
        Target column name.
    alpha : float
        Ridge penalty coefficient. Larger values imply stronger shrinkage.

    Returns
    -------
    RidgeTrainingResult
        Predictions, coefficients, and intercept for the fitted model.
    """
    feature_frame = dataset.loc[:, feature_columns].astype(float)
    target_series = dataset.loc[:, target_column].astype(float)

    standardized_features, _means, _stds = _zscore_frame(feature_frame=feature_frame)
    intercept_column = pd.Series(
        1.0, index=standardized_features.index, name="intercept"
    )
    design_frame = pd.concat([intercept_column, standardized_features], axis=1)
    design_matrix = design_frame.to_numpy(dtype=float)
    target_vector = target_series.to_numpy(dtype=float)

    fitted = _ridge_closed_form(
        design_matrix=design_matrix,
        target_vector=target_vector,
        alpha=float(alpha),
    )
    intercept = float(fitted[0])
    coefficients = fitted[1:]

    prediction = design_matrix @ fitted
    prediction_frame = pd.DataFrame(
        {
            "prediction": prediction,
            "actual": target_vector,
        },
        index=dataset.index,
    )
    coefficient_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": coefficients,
        }
    )
    return RidgeTrainingResult(
        predictions=prediction_frame,
        coefficients=coefficient_frame,
        intercept=intercept,
    )
