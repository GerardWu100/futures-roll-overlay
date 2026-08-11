"""Generate the technical figures used by the bilingual blog post.

The script reads only frozen outputs under ``blog/data``. This keeps the article
reproducible even when later pipeline runs use different dates or settings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLOG_ROOT = Path(__file__).resolve().parent
DATA_DIR = BLOG_ROOT / "data"
IMAGE_DIR = BLOG_ROOT / "images"
FIGURE_DPI = 220
MODEL_ORDER = ["persistence", "ridge"]
ASSET_ORDER = ["ES", "CL", "GC"]
MODEL_LABELS = {"persistence": "Persistence", "ridge": "Ridge"}
ASSET_COLORS = {"ES": "#315C75", "CL": "#C56A32", "GC": "#C8A33D"}


def plot_asset_rmse(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot mean fold root mean squared error by asset and model.

    Parameters
    ----------
    metrics : pd.DataFrame
        Frozen model summary with ``scope``, ``asset``, ``model``, and ``rmse``
        columns. RMSE is measured in annualized variance decimal units.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The function writes a PNG and does not return an object.
    """
    asset_metrics = metrics.loc[metrics["scope"] == "asset"].copy()
    pivoted = asset_metrics.pivot(index="asset", columns="model", values="rmse")
    pivoted = pivoted.loc[ASSET_ORDER, MODEL_ORDER]

    x_positions = np.arange(len(ASSET_ORDER), dtype=float)
    bar_width = 0.34
    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)

    for model_index, model_name in enumerate(MODEL_ORDER):
        offsets = x_positions + (model_index - 0.5) * bar_width
        values = pivoted[model_name].to_numpy(dtype=float)
        bars = axis.bar(
            offsets,
            values,
            width=bar_width,
            label=MODEL_LABELS[model_name],
            color="#315C75" if model_name == "persistence" else "#C56A32",
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)

    axis.set_title("No Forecast Wins Across All Three Futures Markets")
    axis.set_xlabel("Futures root")
    axis.set_ylabel("Mean fold RMSE (annualized variance, decimal)")
    axis.set_xticks(x_positions, ASSET_ORDER)
    axis.set_ylim(0.0, float(pivoted.to_numpy().max()) * 1.22)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)

    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_prediction_diagnostics(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare realized and predicted forward variance for both models.

    Parameters
    ----------
    predictions : pd.DataFrame
        Frozen out-of-sample rows with ``actual``, ``prediction``, ``asset``,
        and ``model`` columns. Each point is one forecast observation.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The function writes a PNG and does not return an object.
    """
    clean = predictions.dropna(subset=["actual", "prediction", "asset", "model"])
    overall_min = float(min(clean["actual"].min(), clean["prediction"].min()))
    overall_max = float(max(clean["actual"].max(), clean["prediction"].max()))
    padding = 0.02
    lower_bound = overall_min - padding
    upper_bound = overall_max + padding

    figure, axes = plt.subplots(
        ncols=2,
        figsize=(13.5, 5.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, model_name in zip(axes, MODEL_ORDER, strict=True):
        model_rows = clean.loc[clean["model"] == model_name]
        for asset_name in ASSET_ORDER:
            asset_rows = model_rows.loc[model_rows["asset"] == asset_name]
            axis.scatter(
                asset_rows["actual"],
                asset_rows["prediction"],
                s=20,
                alpha=0.52,
                color=ASSET_COLORS[asset_name],
                label=asset_name,
                edgecolors="none",
            )
        axis.plot(
            [lower_bound, upper_bound],
            [lower_bound, upper_bound],
            color="#434343",
            linewidth=1.2,
            linestyle="--",
            label="Perfect forecast",
        )
        axis.axhline(0.0, color="#7A7A7A", linewidth=0.8, alpha=0.7)
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xlabel("Realized two-day variance (annualized, decimal)")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Predicted two-day variance (annualized, decimal)")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    figure.suptitle("Out-of-Sample Forecasts Miss the Largest Variance Spikes", y=1.03)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load frozen results and regenerate every technical blog figure."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(DATA_DIR / "metrics.csv")
    predictions = pd.read_parquet(DATA_DIR / "predictions.parquet")

    plot_asset_rmse(metrics=metrics, output_path=IMAGE_DIR / "01_asset_rmse.png")
    plot_prediction_diagnostics(
        predictions=predictions,
        output_path=IMAGE_DIR / "02_prediction_diagnostics.png",
    )


if __name__ == "__main__":
    main()
