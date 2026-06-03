# Part 1: Conceptual Explanation

`src/` follows the standard Python `src` layout. Importable product code lives in `futures_roll_overlay/`, which mirrors the research pipeline directly.

- `data_access/`: raw input boundary and optional refresh tooling.
- `continuous_futures/`: contract stitching into continuous series.
- `features/`: target and feature construction.
- `models/`: baseline and ridge training logic.
- `evaluation/`: metrics and walk-forward evaluation.
- `pipeline/`: end-to-end orchestration and artifact writing.

This structure is deliberate: each layer answers one clear question about the workflow, from data provenance to final forecast interpretation.

# Part 2: Code Reference

- `futures_roll_overlay/data_access/raw_cache.py`: load local futures/roll calendars and manifest inventory.
- `futures_roll_overlay/data_access/refresh_clickhouse_raw.py`: optional ClickHouse refresh command for rebuilding `data/raw`.
- `futures_roll_overlay/continuous_futures/build.py`: `build_continuous(...)` and `ContinuousFutures` dataclass.
- `futures_roll_overlay/features/term_structure.py`: `compute_term_structure_features(...)`.
- `futures_roll_overlay/features/realized_variance.py`: `build_forward_realized_variance(...)`.
- `futures_roll_overlay/features/dataset.py`: `build_asset_dataset(...)` and `build_pooled_dataset(...)`.
- `futures_roll_overlay/models/baselines.py`: persistence and rolling-mean baselines.
- `futures_roll_overlay/models/train.py`: ridge regression fit and coefficient outputs.
- `futures_roll_overlay/evaluation/metrics.py`: `compute_regression_metrics(...)`.
- `futures_roll_overlay/evaluation/walk_forward.py`: fold generation and walk-forward ridge execution.
- `futures_roll_overlay/pipeline/run_research_pipeline.py`: CLI + programmatic entrypoint for full offline run.

Console entrypoints (see `pyproject.toml`): `futures-roll-research`, `futures-roll-refresh-raw`.

# Part 3: Short Journal

- 2026-04-19: Replaced old overlay/backtest stack with a focused realized-variance forecasting architecture.
- 2026-04-19: Added per-asset walk-forward evaluation path to keep index semantics clean and interpretation simple.
- 2026-04-19: Added notebook-specific subset artifact outputs so educational notebook flow stays lightweight without duplicating production logic.
- 2026-05-20: Moved importable code under `futures_roll_overlay/` per standard `src/package_name` layout; exposed CLI via `[project.scripts]`.
