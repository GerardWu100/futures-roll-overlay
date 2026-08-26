# futures-roll-overlay

Offline quantitative research pipeline that asks one question: can futures
term-structure and lagged realized-volatility features forecast forward
realized variance for a small cross-asset futures set? The scope is
intentionally narrow — no options-implied overlays, no portfolio strategy
layer, no Greeks, no dashboards.

## What it does

1. Loads local futures and roll-calendar Parquet files from `data/raw/`.
2. Builds continuous futures series with configurable roll and price-adjustment
   rules (calendar or volume roll; ratio or Panama adjustment).
3. Computes daily log returns and a forward annualized realized-variance
   target.
4. Engineers explainable features from term structure (front-second spread,
   annualized roll yield, slope) and lagged realized variance.
5. Trains two models: an observable trailing-variance persistence baseline,
   and a ridge regression model.
6. Evaluates out-of-sample performance per asset with horizon-purged
   walk-forward splits.
7. Writes run artifacts for interpretation and notebook teaching.

Default assets are ES, CL, and GC (`config.toml`, `[research].assets`).

## Requirements

- Python >= 3.13
- Default runs are fully offline and need no external service or `.env` file.
- Optional: a ClickHouse instance, only for the one-time raw-data refresh
  command (`futures-roll-refresh-raw`). It reads `CLICKHOUSE_HOST`,
  `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`,
  `CLICKHOUSE_SECURE`, `CLICKHOUSE_VERIFY` from a local `.env` file
  (path set by `[raw_refresh].clickhouse_env` in `config.toml`).

## Setup

```bash
uv sync --all-groups
```

`--all-groups` pulls in the `dev` group (pytest, nbconvert, jupyter), needed
for tests and the notebook. `uv sync` alone is enough to run the pipeline.

## Usage

```bash
uv run futures-roll-research
```

Equivalent module invocation:

```bash
uv run python -m futures_roll_overlay.pipeline.run_research_pipeline
```

Optional shell wrapper: `./scripts/run_research_pipeline.sh`.

Optional one-time raw-cache refresh from ClickHouse (writes new Parquet into
`data/raw`; not required for default runs): `uv run futures-roll-refresh-raw`.

Run the tests: `uv run python -m pytest -q`.

Execute the teaching notebook end to end:

```bash
uv run --group dev python -m nbconvert --to notebook --execute \
    notebooks/research_pipeline_demo.ipynb --output executed.ipynb
```

## Configuration

All tunable settings live in `config.toml`:

- `[raw_data]`: local data directories and manifest path.
- `[research]`: assets, date window, roll method/adjustment, target horizon,
  and feature lags.
- `[models]`: ridge regularization parameter.
- `[evaluation]`: walk-forward train/test/step sizes; the target horizon
  determines the training-label purge automatically.
- `[raw_refresh]`: path to the ClickHouse credential file used only by the
  optional refresh command.

## Layout

```text
config.toml              tunable settings for data, research, model, evaluation
data/raw/                tracked offline futures, roll-calendar, and manifest inputs
docs/reference/          methodology notes (realized variance, evaluation protocol, raw-cache contract)
notebooks/                teaching notebook that runs the full pipeline
scripts/                  thin shell wrapper around the CLI entrypoint
src/futures_roll_overlay/ data_access, continuous_futures, features, models, evaluation, pipeline
tests/                    unit tests per stage plus one integration test
outputs/                  run artifacts (not tracked in git)
```

See `GUIDE_ROOT.md` and `GUIDE_OVERVIEW.md` for the domain-logic derivation
and per-folder guides (`GUIDE_<folder>.md`) for module-level detail.

## Output

Each run writes to `outputs/runs/<run_id>/`:

```text
dataset.parquet               assembled feature/target dataset
predictions.parquet           walk-forward out-of-sample predictions
metrics.csv                   pooled and per-asset regression metrics
feature_importance.csv        ridge coefficients
prediction_diagnostics.png    diagnostic plot
```
