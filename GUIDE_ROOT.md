# Part 1: Conceptual Explanation

This repository is an offline-first quantitative research pipeline centered on one question:

**Can futures term-structure and lagged volatility features forecast forward realized variance across a small cross-asset futures set?**

The design is intentionally narrow and interview-defensible. The codebase starts from local raw Parquet files, builds continuous futures series, constructs realized-variance targets and explainable features, trains a baseline and ridge model, evaluates out-of-sample performance with walk-forward splits, and exports compact artifacts for interpretation and notebook teaching.

By default, runtime does not require `.env` or ClickHouse. Database access exists only in an explicit optional raw-refresh command.

# Part 2: Code Reference

- `config.toml`: all tunable settings for raw data paths, research window, model parameters, and evaluation protocol.
- `pyproject.toml`: project metadata, dependencies (`uv`, pytest, Ruff, notebook tools), and CLI entrypoints.
- `.gitignore`: keeps generated outputs ignored while tracking canonical `data/raw` files.
- `data/raw/futures/`: per-asset futures Parquet inputs.
- `data/raw/roll_calendars/`: per-asset sample and full roll calendars.
- `data/raw/manifest/dataset_manifest.json`: machine-readable raw-data contract.
- `src/futures_roll_overlay/data_access/raw_cache.py`: offline loaders and manifest inventory.
- `src/futures_roll_overlay/data_access/refresh_clickhouse_raw.py`: optional one-time ClickHouse refresh into `data/raw`.
- `src/futures_roll_overlay/continuous_futures/build.py`: continuous futures construction and roll-gap adjustment.
- `src/futures_roll_overlay/features/`: term structure, realized variance, and dataset assembly modules.
- `src/futures_roll_overlay/models/`: naive baselines and ridge training.
- `src/futures_roll_overlay/evaluation/`: metrics and walk-forward evaluation.
- `src/futures_roll_overlay/pipeline/run_research_pipeline.py`: end-to-end offline pipeline entrypoint.
- `scripts/run_research_pipeline.sh`: thin shell wrapper around `futures-roll-research`.
- `notebooks/research_pipeline_demo.ipynb`: strict alternating Markdown/code notebook that explains the full pipeline with compact subset artifacts.
- `tests/unit/`: focused unit tests by pipeline stage.
- `tests/integration/`: end-to-end pipeline integration test.

# Part 3: Short Journal

- 2026-04-19: Reframed repository scope from options-overlay portfolio system to a compact realized-variance forecasting pipeline.
- 2026-04-19: Replaced flat `src/` layout with stage-aligned subpackages (`data_access`, `continuous_futures`, `features`, `models`, `evaluation`, `pipeline`).
- 2026-04-19: Migrated raw-data contract to canonical `data/raw/{futures,roll_calendars,manifest}` and removed dependency on `outputs/cache` for default runs.
- 2026-04-19: Added per-asset walk-forward evaluation to keep indexing simple and avoid pooled duplicate-date ambiguity.
- 2026-04-19: Rebuilt notebook as a full teaching artifact with compact subset outputs so execution stays clear and lightweight.
- 2026-05-20: Adopted standard `src/futures_roll_overlay/` package layout, `tests/unit` + `tests/integration`, CLI scripts, and removed obsolete superpowers planning docs.
