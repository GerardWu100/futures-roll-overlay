"""Offline futures-to-forecast research pipeline entrypoint.

This module is the main production orchestration path for the refactored
project. It reads local raw Parquet files, builds continuous futures series,
constructs realized-variance targets and explainable features, trains naive and
ridge baselines with walk-forward evaluation, and writes compact run artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from futures_roll_overlay.continuous_futures.build import build_continuous
from futures_roll_overlay.data_access.raw_cache import (
    get_futures_daily,
    get_roll_calendar,
    load_config,
)
from futures_roll_overlay.evaluation.metrics import compute_regression_metrics
from futures_roll_overlay.evaluation.walk_forward import (
    generate_walk_forward_splits,
    run_walk_forward_regression,
)
from futures_roll_overlay.features.dataset import (
    build_asset_dataset,
    build_pooled_dataset,
)
from futures_roll_overlay.features.realized_variance import (
    build_forward_realized_variance,
)
from futures_roll_overlay.features.term_structure import compute_term_structure_features
from futures_roll_overlay.models.baselines import trailing_variance_persistence
from futures_roll_overlay.models.train import fit_ridge_regression

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs"


def _next_run_id(output_root: Path) -> str:
    """Build run id with ``YYYY-MM-DD_NNN`` format."""
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    output_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_root.glob(f"{date_prefix}_*"))
    if not existing:
        return f"{date_prefix}_001"
    latest_suffix = max(int(path.name.split("_")[-1]) for path in existing)
    return f"{date_prefix}_{latest_suffix + 1:03d}"


def _continuous_log_returns(
    futures_daily: pd.DataFrame,
    roll_calendar: pd.DataFrame,
    roll_method: str,
    adjustment: str,
) -> pd.Series:
    """Build a continuous close series and return date-indexed daily log returns."""
    continuous = build_continuous(
        daily_data=futures_daily,
        roll_calendar=roll_calendar,
        method=roll_method,
        adjustment=adjustment,
    )
    close_series = continuous.prices.set_index("date")["close"].sort_index()
    # The first session has no prior close, so it has no return. Dropping it
    # keeps the series honest; filling it with zero would insert a session on
    # which the market is recorded as not having moved, and that fake zero
    # would flow straight into the realized-variance target.
    simple_returns = close_series.pct_change().dropna()
    log_returns = np.log1p(simple_returns)
    log_returns.name = "log_return"
    return log_returns


def _asset_feature_dataset(asset: str, config: dict[str, Any]) -> pd.DataFrame:
    """Build one asset-level modeling dataset from local raw files."""
    start_date = config["research"]["start_date"]
    end_date = config["research"]["end_date"]

    futures_daily = get_futures_daily(
        root=asset,
        start=start_date,
        end=end_date,
        config=config,
    )
    roll_calendar_for_build = get_roll_calendar(
        root=asset,
        config=config,
        start=start_date,
        end=end_date,
        variant=config["research"]["roll_calendar_build_variant"],
    )
    roll_calendar_for_timing = get_roll_calendar(
        root=asset,
        config=config,
        variant=config["research"]["roll_calendar_timing_variant"],
    )

    log_returns = _continuous_log_returns(
        futures_daily=futures_daily,
        roll_calendar=roll_calendar_for_build,
        roll_method=config["research"]["roll_method"],
        adjustment=config["research"]["roll_adjustment"],
    )

    # The newest contract in a roll calendar is still the front contract on the
    # calendar's last date, so its maximum date records where the file stops,
    # not where the contract rolled out. Dropping it sends term-structure
    # timing to the month-code expiry proxy for that contract instead of using
    # a censored date as a maturity-gap denominator.
    contract_end_dates = roll_calendar_for_timing.groupby("contract")["date"].max()
    calendar_last_date = roll_calendar_for_timing["date"].max()
    contract_end_dates = contract_end_dates[contract_end_dates < calendar_last_date]
    term_features = compute_term_structure_features(
        daily_data=futures_daily.loc[:, ["date", "symbol", "close"]],
        contract_end_dates=contract_end_dates,
        flat_threshold=float(config["research"]["flat_threshold"]),
        num_contracts=int(config["research"]["num_contracts"]),
        regime_persistence_days=int(config["research"]["regime_persistence_days"]),
    )
    rv_targets = build_forward_realized_variance(
        log_returns=log_returns,
        horizon_days=int(config["research"]["target_horizon_days"]),
        annualization_factor=int(config["research"]["annualization_factor"]),
    )
    rv_targets = rv_targets.reset_index().rename(columns={"index": "date"})

    dataset = build_asset_dataset(
        asset=asset,
        term_features=term_features,
        realized_variance=rv_targets,
        lag_days=int(config["research"]["lag_days"]),
    )
    return dataset


def _apply_asset_sample_filter(
    dataset: pd.DataFrame,
    min_asset_rows: int,
) -> pd.DataFrame:
    """Keep only assets with enough rows for walk-forward evaluation.

    Parameters
    ----------
    dataset : pd.DataFrame
        Concatenated pooled dataset containing an ``asset`` column.
    min_asset_rows : int
        Minimum required rows per asset to remain in the pooled panel.

    Returns
    -------
    pd.DataFrame
        Filtered dataset with short-history assets removed.
    """
    if dataset.empty:
        return dataset
    row_counts = dataset.groupby("asset").size()
    eligible_assets = row_counts[row_counts >= min_asset_rows].index
    return dataset[dataset["asset"].isin(eligible_assets)].copy()


def _feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return model feature columns from prepared pooled dataset."""
    excluded = {
        "date",
        "asset",
        "target_forward_rv_annualized",
        "forward_realized_variance",
        "forward_realized_variance_annualized",
        "daily_realized_variance",
        "known_trailing_rv_annualized",
        "index",
    }
    return [column for column in dataset.columns if column not in excluded]


def _label_walk_forward_outputs(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    asset_name: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach asset and model identifiers to one walk-forward evaluation result."""
    labeled_metrics = fold_metrics.copy()
    labeled_metrics["asset"] = asset_name
    labeled_metrics["model"] = model_name
    labeled_predictions = predictions.copy()
    labeled_predictions["asset"] = asset_name
    labeled_predictions["model"] = model_name
    return labeled_metrics, labeled_predictions


def _baseline_walk_forward(
    dataset: pd.DataFrame,
    target_column: str,
    trailing_variance_column: str,
    train_size: int,
    test_size: int,
    step_size: int,
    label_horizon_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate feasible trailing-variance persistence over purged folds.

    Parameters
    ----------
    dataset : pd.DataFrame
        Date-indexed modeling panel for one futures root.
    target_column : str
        Forward realized-variance column used as the observed outcome.
    trailing_variance_column : str
        Observable trailing-variance column used as the persistence forecast.
    train_size : int
        Nominal first test-row position before the horizon purge.
    test_size : int
        Number of rows in each out-of-sample block.
    step_size : int
        Number of rows between consecutive test-block starts.
    label_horizon_days : int
        Forward target horizon ``H`` used to purge unavailable labels.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Fold-level regression metrics and observation-level predictions.
    """
    ordered = dataset.sort_index()
    full_persistence = trailing_variance_persistence(
        trailing_variance=ordered[trailing_variance_column]
    )
    splits = generate_walk_forward_splits(
        date_index=ordered.index,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        label_horizon_days=label_horizon_days,
    )
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold_id, split in enumerate(splits, start=1):
        test = ordered.iloc[split.test_start : split.test_end + 1]
        test_prediction = full_persistence.reindex(test.index)
        fold_metrics = compute_regression_metrics(
            actual=test[target_column],
            prediction=test_prediction,
        )
        metric_rows.append(
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
    return pd.DataFrame(metric_rows), predictions


def _mean_fold_metrics(fold_metrics: pd.DataFrame) -> dict[str, float]:
    """Average fold metrics; report missing values when there are no folds."""
    metric_columns = ["mse", "rmse", "mae", "r2"]
    if fold_metrics.empty:
        # Zeros here would look like a perfect model in metrics.csv.
        return {metric_name: float("nan") for metric_name in metric_columns}
    return {
        metric_name: float(fold_metrics[metric_name].mean())
        for metric_name in metric_columns
    }


def _align_asset_datasets(asset_datasets: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Align per-asset datasets onto a shared feature-column set.

    Regime one-hot columns can differ by asset depending on observed states.
    This helper ensures every asset frame has the same feature columns so the
    pooled ridge fit and per-asset evaluation use a consistent design matrix.
    """
    if not asset_datasets:
        return asset_datasets
    non_feature_columns = {
        "date",
        "asset",
        "target_forward_rv_annualized",
        "forward_realized_variance",
        "known_trailing_rv_annualized",
    }
    shared_feature_columns = sorted(
        {
            column_name
            for asset_dataset in asset_datasets
            for column_name in asset_dataset.columns
            if column_name not in non_feature_columns
        }
    )

    aligned: list[pd.DataFrame] = []
    for asset_dataset in asset_datasets:
        frame = asset_dataset.copy()
        for feature_column in shared_feature_columns:
            if feature_column not in frame.columns:
                # Missing regime dummies for an asset are treated as inactive (0).
                frame[feature_column] = 0.0
        ordered_columns = [
            "date",
            *shared_feature_columns,
            "forward_realized_variance",
            "known_trailing_rv_annualized",
            "asset",
            "target_forward_rv_annualized",
        ]
        aligned.append(frame.loc[:, ordered_columns].copy())
    return aligned


def _notebook_asset_dataset_window(
    ordered_asset_dataset: pd.DataFrame,
    asset_predictions: pd.DataFrame,
    max_rows: int,
) -> pd.DataFrame:
    """Pick one asset's date window centered near the first OOS prediction."""
    if asset_predictions.empty:
        return ordered_asset_dataset.head(max_rows)

    first_prediction_date = pd.Timestamp(asset_predictions["date"].min())
    prediction_start_candidates = ordered_asset_dataset.index[
        ordered_asset_dataset["date"] >= first_prediction_date
    ]
    if len(prediction_start_candidates) == 0:
        return ordered_asset_dataset.tail(max_rows)

    prediction_start_index = int(prediction_start_candidates[0])
    history_rows = max_rows // 2
    start_index = max(0, prediction_start_index - history_rows)
    end_index = min(len(ordered_asset_dataset), start_index + max_rows)
    return ordered_asset_dataset.iloc[start_index:end_index].copy()


def _clip_predictions_to_dataset_window(
    asset_predictions: pd.DataFrame,
    asset_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only predictions that fall inside the selected dataset date span."""
    asset_dates = asset_dataset["date"].sort_values().drop_duplicates()
    if asset_dates.empty or asset_predictions.empty:
        return pd.DataFrame(columns=asset_predictions.columns)
    start_date = pd.Timestamp(asset_dates.min())
    end_date = pd.Timestamp(asset_dates.max())
    return asset_predictions[
        (asset_predictions["date"] >= start_date)
        & (asset_predictions["date"] <= end_date)
    ].copy()


def _select_notebook_demo_subset(
    pooled_dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a compact subset for notebook-first demonstration.

    The notebook should explain the full pipeline while staying easy to run and
    read. This helper keeps all assets but truncates to roughly ``max_rows``
    rows per asset in date order, centered on the first out-of-sample prediction.
    """
    if pooled_dataset.empty:
        return pooled_dataset, predictions

    dataset_reset = pooled_dataset.reset_index()
    prediction_frame = predictions.reset_index().rename(columns={"index": "date"})
    prediction_frame["date"] = pd.to_datetime(prediction_frame["date"])

    selected_frames: list[pd.DataFrame] = []
    for asset_name, asset_dataset in dataset_reset.groupby("asset"):
        ordered_asset_dataset = asset_dataset.sort_values("date").reset_index(drop=True)
        asset_predictions = prediction_frame[
            prediction_frame["asset"] == asset_name
        ].sort_values("date")
        selected_frames.append(
            _notebook_asset_dataset_window(
                ordered_asset_dataset=ordered_asset_dataset,
                asset_predictions=asset_predictions,
                max_rows=max_rows,
            )
        )

    selected = pd.concat(selected_frames, axis=0, ignore_index=True)

    notebook_prediction_frames: list[pd.DataFrame] = []
    for asset_name, asset_slice in selected.groupby("asset"):
        asset_predictions = prediction_frame[
            prediction_frame["asset"] == asset_name
        ].copy()
        clipped = _clip_predictions_to_dataset_window(
            asset_predictions=asset_predictions,
            asset_dataset=asset_slice,
        )
        if not clipped.empty:
            notebook_prediction_frames.append(clipped)

    if notebook_prediction_frames:
        filtered_predictions = pd.concat(notebook_prediction_frames, ignore_index=True)
    else:
        filtered_predictions = pd.DataFrame(columns=prediction_frame.columns)
    return selected, filtered_predictions


def _plot_prediction_diagnostics(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save prediction-vs-realized diagnostic scatter plot."""
    figure, axis = plt.subplots(figsize=(10, 6), dpi=180, constrained_layout=True)
    axis.scatter(
        predictions["actual"],
        predictions["prediction"],
        alpha=0.75,
        s=40,
        color="#264653",
        edgecolors="none",
    )
    if len(predictions) > 0:
        min_x = float(min(predictions["actual"].min(), predictions["prediction"].min()))
        max_x = float(max(predictions["actual"].max(), predictions["prediction"].max()))
        axis.plot([min_x, max_x], [min_x, max_x], color="#e76f51", linewidth=1.5)
    axis.set_title("Predicted vs Realized Forward Variance")
    axis.set_xlabel("Realized Forward Variance (annualized, decimal)")
    axis.set_ylabel("Predicted Forward Variance (annualized, decimal)")
    figure.savefig(output_path)
    plt.close(figure)


def run_research_pipeline(
    config_path: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Run the full offline realized-variance research pipeline.

    Parameters
    ----------
    config_path : Path | None, default None
        Optional path to a non-default config TOML file.
    output_root : Path | None, default None
        Optional root directory for run artifacts.

    Returns
    -------
    dict[str, Any]
        Dictionary with run path and key in-memory artifacts.
    """
    config = load_config(config_path=config_path)
    assets = list(config["research"]["assets"])

    asset_datasets = [
        _asset_feature_dataset(asset=asset, config=config) for asset in assets
    ]
    asset_datasets = _align_asset_datasets(asset_datasets=asset_datasets)

    eligible_asset_datasets = [
        _apply_asset_sample_filter(
            dataset=asset_dataset,
            min_asset_rows=int(config["evaluation"]["min_asset_rows"]),
        )
        for asset_dataset in asset_datasets
    ]
    eligible_asset_datasets = [
        asset_dataset
        for asset_dataset in eligible_asset_datasets
        if not asset_dataset.empty
    ]

    if not eligible_asset_datasets:
        raise ValueError(
            "No eligible rows for modeling after asset-history filtering. "
            "Increase date range or reduce evaluation.min_asset_rows."
        )

    per_asset_fold_metrics: list[pd.DataFrame] = []
    per_asset_predictions: list[pd.DataFrame] = []
    per_asset_metric_rows: list[dict[str, object]] = []

    feature_columns = _feature_columns(dataset=eligible_asset_datasets[0])
    target_column = "target_forward_rv_annualized"
    train_size = int(config["evaluation"]["train_size"])
    test_size = int(config["evaluation"]["test_size"])
    step_size = int(config["evaluation"]["step_size"])
    label_horizon_days = int(config["research"]["target_horizon_days"])

    for asset_dataset in eligible_asset_datasets:
        asset_name = str(asset_dataset["asset"].iloc[0])
        ordered_asset_dataset = asset_dataset.sort_values("date").reset_index(drop=True)
        ordered_asset_dataset = ordered_asset_dataset.set_index("date")

        baseline_metrics, baseline_predictions = _baseline_walk_forward(
            dataset=ordered_asset_dataset,
            target_column=target_column,
            trailing_variance_column="known_trailing_rv_annualized",
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            label_horizon_days=label_horizon_days,
        )
        baseline_metrics, baseline_predictions = _label_walk_forward_outputs(
            baseline_metrics,
            baseline_predictions,
            asset_name=asset_name,
            model_name="persistence",
        )

        ridge_metrics, ridge_predictions = run_walk_forward_regression(
            dataset=ordered_asset_dataset,
            feature_columns=feature_columns,
            target_column=target_column,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            label_horizon_days=label_horizon_days,
            alpha=float(config["models"]["ridge_alpha"]),
        )
        ridge_metrics, ridge_predictions = _label_walk_forward_outputs(
            ridge_metrics,
            ridge_predictions,
            asset_name=asset_name,
            model_name="ridge",
        )

        per_asset_fold_metrics.extend([baseline_metrics, ridge_metrics])
        per_asset_predictions.extend([baseline_predictions, ridge_predictions])

        per_asset_metric_rows.extend(
            [
                {
                    "model": "persistence",
                    "asset": asset_name,
                    **_mean_fold_metrics(fold_metrics=baseline_metrics),
                },
                {
                    "model": "ridge",
                    "asset": asset_name,
                    **_mean_fold_metrics(fold_metrics=ridge_metrics),
                },
            ]
        )

    all_fold_metrics = pd.concat(per_asset_fold_metrics, axis=0, ignore_index=True)
    all_predictions = pd.concat(per_asset_predictions, axis=0)
    all_predictions = all_predictions.sort_index()

    pooled_metric_rows: list[dict[str, object]] = []
    for model_name in ["persistence", "ridge"]:
        model_fold_metrics = all_fold_metrics[all_fold_metrics["model"] == model_name]
        pooled_metric_rows.append(
            {
                "model": model_name,
                **_mean_fold_metrics(fold_metrics=model_fold_metrics),
            }
        )
    metric_summary = pd.DataFrame(pooled_metric_rows)

    per_asset_summary = pd.DataFrame(per_asset_metric_rows)

    pooled_dataset = build_pooled_dataset(asset_datasets=eligible_asset_datasets)
    pooled_dataset = pooled_dataset.sort_values(["date", "asset"]).reset_index(
        drop=True
    )
    pooled_dataset = pooled_dataset.set_index("date")

    full_fit = fit_ridge_regression(
        dataset=pooled_dataset,
        feature_columns=feature_columns,
        target_column=target_column,
        alpha=float(config["models"]["ridge_alpha"]),
    )

    resolved_output_root = output_root or DEFAULT_OUTPUT_ROOT
    run_id = _next_run_id(output_root=resolved_output_root)
    run_dir = resolved_output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_to_save = pooled_dataset.reset_index()
    notebook_dataset, notebook_predictions = _select_notebook_demo_subset(
        pooled_dataset=pooled_dataset,
        predictions=all_predictions,
        max_rows=int(config["evaluation"]["notebook_max_rows_per_asset"]),
    )
    dataset_to_save.to_parquet(run_dir / "dataset.parquet", index=False)
    all_predictions.reset_index().to_parquet(
        run_dir / "predictions.parquet", index=False
    )
    notebook_dataset.to_parquet(run_dir / "notebook_dataset.parquet", index=False)
    notebook_predictions.to_parquet(
        run_dir / "notebook_predictions.parquet", index=False
    )
    metric_output = pd.concat(
        [
            metric_summary.assign(scope="pooled"),
            per_asset_summary.assign(scope="asset"),
        ],
        axis=0,
        ignore_index=True,
    )
    metric_output.to_csv(run_dir / "metrics.csv", index=False)
    full_fit.coefficients.to_csv(run_dir / "feature_importance.csv", index=False)
    _plot_prediction_diagnostics(
        predictions=all_predictions.dropna(subset=["prediction", "actual"]),
        output_path=run_dir / "prediction_diagnostics.png",
    )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "dataset": pooled_dataset,
        "predictions": all_predictions,
        "metrics": metric_output,
        "feature_importance": full_fit.coefficients,
        "notebook_dataset": notebook_dataset,
        "notebook_predictions": notebook_predictions,
    }


def _parse_args() -> argparse.Namespace:
    """Parse pipeline CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the offline futures realized-variance research pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to alternate config.toml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root for run artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for offline pipeline execution."""
    args = _parse_args()
    result = run_research_pipeline(
        config_path=args.config, output_root=args.output_root
    )
    print(f"run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
