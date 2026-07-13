"""Rebuild blog evidence with a correctly aligned forward-variance target.

The production target currently uses ``shift(-1).rolling(H).sum()``. Rolling
windows look backward, so that expression includes the return at the forecast
date when ``H`` is greater than one. This audit script changes only the target
alignment and reuses the project's remaining pipeline and evaluation logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from futures_roll_overlay.data_access.raw_cache import (
    get_futures_daily,
    get_roll_calendar,
    load_config,
)
from futures_roll_overlay.evaluation.metrics import compute_regression_metrics
from futures_roll_overlay.evaluation.walk_forward import (
    _ridge_test_predictions,
    generate_walk_forward_splits,
)
from futures_roll_overlay.features.dataset import build_asset_dataset
from futures_roll_overlay.features.term_structure import compute_term_structure_features
from futures_roll_overlay.pipeline.run_research_pipeline import (
    _align_asset_datasets,
    _continuous_log_returns,
    _feature_columns,
    _mean_fold_metrics,
)


BLOG_ROOT = Path(__file__).resolve().parent
DATA_DIR = BLOG_ROOT / "data"
TARGET_COLUMN = "target_forward_rv_annualized"


def build_corrected_target(
    log_returns: pd.Series,
    horizon_days: int,
    annualization_factor: int,
) -> pd.DataFrame:
    """Construct variance from the next ``H`` returns, excluding date ``t``.

    Parameters
    ----------
    log_returns : pd.Series
        Date-indexed daily log returns ``r_t`` in decimal units.
    horizon_days : int
        Number of future sessions ``H`` included in each target.
    annualization_factor : int
        Sessions per year ``A`` used to annualize the horizon sum.

    Returns
    -------
    pd.DataFrame
        Date-indexed target frame. At date ``t``, ``forward_realized_variance``
        is ``sum(r_{t+i}^2 for i in 1..H)``.
    """
    squared_returns = log_returns.pow(2)
    future_terms = [squared_returns.shift(-lead) for lead in range(1, horizon_days + 1)]
    forward_variance = pd.concat(future_terms, axis=1).sum(axis=1, min_count=horizon_days)
    return pd.DataFrame(
        {
            "log_return": log_returns,
            "daily_realized_variance": squared_returns,
            "known_trailing_rv_annualized": (
                annualization_factor / horizon_days
            )
            * squared_returns.rolling(
                window=horizon_days,
                min_periods=horizon_days,
            ).sum(),
            "forward_realized_variance": forward_variance,
            "forward_realized_variance_annualized": (
                annualization_factor / horizon_days
            )
            * forward_variance,
        }
    )


def build_corrected_asset_dataset(
    asset: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build one model panel with the corrected future-only target.

    Parameters
    ----------
    asset : str
        Futures root, such as ``ES``, ``CL``, or ``GC``.
    config : dict[str, Any]
        Parsed project configuration.

    Returns
    -------
    pd.DataFrame
        Date-indexed feature and target observations for one futures root.
    """
    research = config["research"]
    daily_data = get_futures_daily(
        root=asset,
        start=research["start_date"],
        end=research["end_date"],
        config=config,
    )
    build_calendar = get_roll_calendar(
        root=asset,
        config=config,
        start=research["start_date"],
        end=research["end_date"],
        variant=research["roll_calendar_build_variant"],
    )
    timing_calendar = get_roll_calendar(
        root=asset,
        config=config,
        variant=research["roll_calendar_timing_variant"],
    )
    log_returns = _continuous_log_returns(
        futures_daily=daily_data,
        roll_calendar=build_calendar,
        roll_method=research["roll_method"],
        adjustment=research["roll_adjustment"],
    )
    corrected_target = build_corrected_target(
        log_returns=log_returns,
        horizon_days=int(research["target_horizon_days"]),
        annualization_factor=int(research["annualization_factor"]),
    ).reset_index()
    contract_end_dates = timing_calendar.groupby("contract")["date"].max()
    term_features = compute_term_structure_features(
        daily_data=daily_data.loc[:, ["date", "symbol", "close"]],
        contract_end_dates=contract_end_dates,
        flat_threshold=float(research["flat_threshold"]),
        num_contracts=int(research["num_contracts"]),
        regime_persistence_days=int(research["regime_persistence_days"]),
    )
    return build_asset_dataset(
        asset=asset,
        term_features=term_features,
        realized_variance=corrected_target,
        lag_days=int(research["lag_days"]),
    )


def evaluate_corrected_datasets(
    asset_datasets: list[pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the original walk-forward models on corrected target panels.

    Parameters
    ----------
    asset_datasets : list[pd.DataFrame]
        One corrected feature panel per futures root.
    config : dict[str, Any]
        Parsed model and evaluation settings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Summary metrics and observation-level out-of-sample predictions.
    """
    evaluation = config["evaluation"]
    horizon_days = int(config["research"]["target_horizon_days"])
    feature_columns = [
        feature_name
        for feature_name in _feature_columns(asset_datasets[0])
        if feature_name != "known_trailing_rv_annualized"
    ]
    metric_rows: list[dict[str, object]] = []
    fold_metrics_by_model: dict[str, list[pd.DataFrame]] = {
        "persistence": [],
        "ridge": [],
    }
    prediction_frames: list[pd.DataFrame] = []

    for asset_dataset in asset_datasets:
        asset_name = str(asset_dataset["asset"].iloc[0])
        ordered = asset_dataset.sort_values("date").set_index("date")
        splits = generate_walk_forward_splits(
            date_index=ordered.index,
            train_size=int(evaluation["train_size"]),
            test_size=int(evaluation["test_size"]),
            step_size=int(evaluation["step_size"]),
        )
        fold_metric_rows = {"persistence": [], "ridge": []}
        model_prediction_rows: dict[str, list[pd.DataFrame]] = {
            "persistence": [],
            "ridge": [],
        }
        for fold_id, split in enumerate(splits, start=1):
            # At the first test date t, only labels ending by t are observable.
            # A target starting at s ends at s + H, so the latest usable row is
            # s = t - H. Relative to the ordinary split, purge H - 1 rows.
            purged_train_end = split.train_end - (horizon_days - 1)
            train = ordered.iloc[split.train_start : purged_train_end + 1]
            test = ordered.iloc[split.test_start : split.test_end + 1]
            fold_predictions = {
                "persistence": test["known_trailing_rv_annualized"].rename(
                    "prediction"
                ),
                "ridge": _ridge_test_predictions(
                    train=train,
                    test=test,
                    feature_columns=feature_columns,
                    target_column=TARGET_COLUMN,
                    alpha=float(config["models"]["ridge_alpha"]),
                ),
            }
            for model_name, prediction in fold_predictions.items():
                fold_metric_rows[model_name].append(
                    {
                        "fold": fold_id,
                        **compute_regression_metrics(
                            actual=test[TARGET_COLUMN],
                            prediction=prediction,
                        ),
                    }
                )
                model_prediction_rows[model_name].append(
                    pd.DataFrame(
                        {
                            "date": test.index,
                            "fold": fold_id,
                            "asset": asset_name,
                            "prediction": prediction.to_numpy(),
                            "actual": test[TARGET_COLUMN].to_numpy(),
                        }
                    )
                )

        for model_name, fold_metrics, predictions in (
            (
                "persistence",
                pd.DataFrame(fold_metric_rows["persistence"]),
                pd.concat(model_prediction_rows["persistence"], ignore_index=True),
            ),
            (
                "ridge",
                pd.DataFrame(fold_metric_rows["ridge"]),
                pd.concat(model_prediction_rows["ridge"], ignore_index=True),
            ),
        ):
            fold_metrics_by_model[model_name].append(fold_metrics)
            metric_rows.append(
                {
                    "model": model_name,
                    "asset": asset_name,
                    **_mean_fold_metrics(fold_metrics),
                    "scope": "asset",
                }
            )
            labeled_predictions = predictions.copy()
            labeled_predictions["asset"] = asset_name
            labeled_predictions["model"] = model_name
            prediction_frames.append(labeled_predictions)

    pooled_rows: list[dict[str, object]] = []
    for model_name, model_frames in fold_metrics_by_model.items():
        pooled_fold_metrics = pd.concat(model_frames, ignore_index=True)
        pooled_rows.append(
            {
                "model": model_name,
                **_mean_fold_metrics(pooled_fold_metrics),
                "scope": "pooled",
                "asset": "",
            }
        )
    metrics = pd.DataFrame([*pooled_rows, *metric_rows])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions


def main() -> None:
    """Build and freeze corrected metrics and predictions under ``blog/data``."""
    config = load_config()
    raw_asset_datasets = [
        build_corrected_asset_dataset(asset=asset, config=config)
        for asset in config["research"]["assets"]
    ]
    asset_datasets = _align_asset_datasets(raw_asset_datasets)
    metrics, predictions = evaluate_corrected_datasets(
        asset_datasets=asset_datasets,
        config=config,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DATA_DIR / "corrected_metrics.csv", index=False)
    predictions.to_parquet(DATA_DIR / "corrected_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
