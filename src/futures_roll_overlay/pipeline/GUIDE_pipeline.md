# Pipeline Guide

This folder wires all layers into a single reproducible workflow.

- `run_research_pipeline.py`:
  - loads config,
  - builds per-asset datasets,
  - runs per-asset horizon-purged walk-forward evaluation,
  - aggregates pooled and per-asset metrics,
  - writes run artifacts under `outputs/runs/<run_id>/`.

Artifacts written by default:

- `dataset.parquet`
- `predictions.parquet`
- `metrics.csv`
- `feature_importance.csv`
- `prediction_diagnostics.png`
- `notebook_dataset.parquet`
- `notebook_predictions.parquet`

Interview story for this layer:

1. One command runs from raw data to interpretable outputs.
2. Output naming and layout are stable.
3. Notebook can consume compact subset artifacts without reimplementing logic.
