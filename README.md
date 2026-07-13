# Portable Futures Research Backbone

This repository is a small offline-first quantitative finance research project.

Core research question:

**Can futures term-structure and lagged volatility features help forecast
forward realized variance for a small cross-asset futures set?**

The project is intentionally narrow. It does **not** include options-implied
overlays, portfolio strategy products, Greeks, stress dashboards, or HTML
reporting surfaces.

## What This Project Does

1. Loads local futures and roll-calendar Parquet files from `data/raw/`.
2. Builds continuous futures series with configurable roll and adjustment rules.
3. Computes daily log returns and forward realized-variance targets.
4. Engineers explainable features from term structure and lagged variance.
5. Trains two forecast baselines:
   - observable trailing-variance persistence baseline,
   - ridge regression model.
6. Evaluates out-of-sample performance with horizon-purged walk-forward splits.
7. Writes compact run artifacts for interpretation and discussion.

## Raw Data Contract (Offline Default)

Default runs read only from `data/raw`.

Required layout:

```text
data/raw/
├── futures/
│   ├── ES.parquet
│   ├── CL.parquet
│   └── GC.parquet
├── roll_calendars/
│   ├── ES_sample.parquet
│   ├── ES_full.parquet
│   ├── CL_sample.parquet
│   ├── CL_full.parquet
│   ├── GC_sample.parquet
│   └── GC_full.parquet
└── manifest/
    └── dataset_manifest.json
```

The manifest records asset, table, date coverage, row counts, schema columns,
and file sizes.

## Configuration

All tunable settings live in `config.toml`.

Main sections:

- `[raw_data]`: local data directories and manifest path.
- `[research]`: date window, roll settings, target horizon, and feature lags.
- `[models]`: ridge regularization parameter.
- `[evaluation]`: nominal walk-forward boundary, test length, and step length;
  the target horizon determines the training-label purge automatically.

## Run Offline Pipeline

```bash
uv sync --all-groups
uv run futures-roll-research
```

Equivalent module invocation:

```bash
uv run python -m futures_roll_overlay.pipeline.run_research_pipeline
```

Output convention:

```text
outputs/runs/<run_id>/
├── dataset.parquet
├── predictions.parquet
├── metrics.csv
├── feature_importance.csv
└── prediction_diagnostics.png
```

## Execute Teaching Notebook

```bash
uv sync --all-groups
uv run --group dev python -m nbconvert --to notebook --execute notebooks/research_pipeline_demo.ipynb --output executed.ipynb
```

The notebook follows strict Markdown/code alternation and demonstrates each
pipeline stage directly from local Parquet inputs.

## Tests And Lint

```bash
uv run python -m pytest -q
uv run ruff check src tests
```

Shell wrapper (optional):

```bash
./scripts/run_research_pipeline.sh
```

## Optional One-Time ClickHouse Refresh (Not Default Path)

ClickHouse can be used only for explicit raw-cache refresh workflows that write
new Parquet bundles into `data/raw`. Default tests, pipeline runs, and notebook
execution do not require `.env` or database access.
