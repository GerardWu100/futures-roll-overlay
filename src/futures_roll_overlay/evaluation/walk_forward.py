"""Walk-forward split generation and out-of-sample evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from futures_roll_overlay.evaluation.metrics import compute_regression_metrics
from futures_roll_overlay.models.train import apply_feature_standardizer
from futures_roll_overlay.models.train import fit_feature_standardizer
from futures_roll_overlay.models.train import fit_ridge_regression


@dataclass(frozen=True)
class WalkForwardSplit:
    """One walk-forward train/test index window.

    Attributes
    ----------
    train_start : int
        Integer position of training window start.
    train_end : int
        Integer position of training window end (inclusive).
    test_start : int
        Integer position of test window start.
    test_end : int
        Integer position of test window end (inclusive).
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def generate_walk_forward_splits(
    date_index: pd.Index,
    train_size: int,
    test_size: int,
    step_size: int,
    label_horizon_days: int,
) -> list[WalkForwardSplit]:
    """Generate ordered folds whose training labels are observable at test time.

    A label dated ``s`` contains returns through ``s + H``, where ``H`` is
    ``label_horizon_days``. If the first test row is ``T``, the latest allowed
    training row is therefore ``T - H``. The omitted ``H - 1`` rows form the
    horizon purge between the training and test slices.

    Parameters
    ----------
    date_index : pd.Index
        Ordered observation dates or labels used to determine the row count.
    train_size : int
        Integer position of the first test row. This is the nominal training
        boundary before the horizon purge.
    test_size : int
        Number of consecutive rows in each test fold.
    step_size : int
        Number of rows to advance the first test position between folds.
    label_horizon_days : int
        Number of future sessions ``H`` included in each training label.

    Returns
    -------
    list[WalkForwardSplit]
        Ordered inclusive row boundaries. Each ``train_end`` is exactly
        ``label_horizon_days`` rows before its corresponding ``test_start``.
    """
    if min(train_size, test_size, step_size, label_horizon_days) < 1:
        raise ValueError("Split sizes and label_horizon_days must be at least 1.")
    if label_horizon_days > train_size:
        raise ValueError("label_horizon_days cannot exceed train_size.")
    row_count = len(date_index)
    splits: list[WalkForwardSplit] = []
    train_start = 0
    test_start = train_size
    while True:
        train_end = test_start - label_horizon_days
        test_end = test_start + test_size - 1
        if test_end >= row_count:
            break
        splits.append(
            WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        test_start = test_start + step_size
    return splits


def _ridge_test_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    alpha: float,
) -> pd.Series:
    """Fit ridge on one fold's train slice and predict the aligned test slice."""
    model = fit_ridge_regression(
        dataset=train,
        feature_columns=feature_columns,
        target_column=target_column,
        alpha=alpha,
    )
    train_features = train.loc[:, feature_columns]
    test_features = test.loc[:, feature_columns]
    train_means, train_stds = fit_feature_standardizer(feature_frame=train_features)
    standardized_test = apply_feature_standardizer(
        feature_frame=test_features,
        means=train_means,
        stds=train_stds,
    )
    coefficient_vector = model.coefficients["coefficient"].to_numpy()
    prediction_values = (
        model.intercept + standardized_test.to_numpy() @ coefficient_vector
    )
    return pd.Series(prediction_values, index=test.index, name="prediction")


def run_walk_forward_regression(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    train_size: int,
    test_size: int,
    step_size: int,
    label_horizon_days: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate ridge regression with walk-forward out-of-sample slices.

    Parameters
    ----------
    dataset : pd.DataFrame
        Full date-indexed modeling panel.
    feature_columns : list[str]
        Feature names passed to ridge training.
    target_column : str
        Target column name.
    train_size : int
        Integer position of the first test row before the horizon purge.
    test_size : int
        Number of rows in each out-of-sample test fold.
    step_size : int
        Rows to move training endpoint forward between folds.
    label_horizon_days : int
        Number of future sessions in each label. The evaluator purges the
        final ``H - 1`` nominal training rows so labels are known at test time.
    alpha : float
        Ridge regularization strength.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(fold_metrics, predictions)`` where ``fold_metrics`` has one row per
        fold, and ``predictions`` has one row per out-of-sample date.
    """
    ordered = dataset.sort_index()
    splits = generate_walk_forward_splits(
        date_index=ordered.index,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        label_horizon_days=label_horizon_days,
    )

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold_id, split in enumerate(splits, start=1):
        train = ordered.iloc[split.train_start : split.train_end + 1]
        test = ordered.iloc[split.test_start : split.test_end + 1]
        test_prediction = _ridge_test_predictions(
            train=train,
            test=test,
            feature_columns=feature_columns,
            target_column=target_column,
            alpha=alpha,
        )
        fold_metrics = compute_regression_metrics(
            actual=test[target_column],
            prediction=test_prediction,
        )
        metrics_rows.append(
            {
                "fold": fold_id,
                "train_start": ordered.index[split.train_start],
                "train_end": ordered.index[split.train_end],
                "test_start": ordered.index[split.test_start],
                "test_end": ordered.index[split.test_end],
                **fold_metrics,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "fold": fold_id,
                    "asset": test["asset"],
                    "prediction": test_prediction,
                    "actual": test[target_column],
                }
            )
        )

    if prediction_rows:
        predictions = pd.concat(prediction_rows, axis=0)
    else:
        predictions = pd.DataFrame(columns=["fold", "asset", "prediction", "actual"])

    return pd.DataFrame(metrics_rows), predictions
