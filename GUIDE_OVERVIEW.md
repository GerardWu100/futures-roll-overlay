# Project Overview

## File Tree

```text
futures-roll-overlay/
├── config.toml
├── data/
│   └── raw/
│       ├── futures/
│       ├── roll_calendars/
│       └── manifest/
├── docs/
│   └── reference/
├── notebooks/
│   └── research_pipeline_demo.ipynb
├── scripts/
│   └── run_research_pipeline.sh
├── src/
│   ├── GUIDE_src.md
│   └── futures_roll_overlay/
│       ├── data_access/
│       ├── continuous_futures/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       └── pipeline/
└── tests/
    ├── unit/
    │   ├── data_access/
    │   ├── continuous_futures/
    │   ├── features/
    │   ├── models/
    │   └── evaluation/
    └── integration/
        └── test_pipeline_integration.py
```

## Pipeline

1. **Data access:** load local futures and roll-calendar Parquet files, and read manifest inventory.
2. **Continuous series:** build front-contract continuous futures with explicit roll and adjustment configuration.
3. **Target construction:** compute daily log returns, daily realized variance, and forward annualized realized variance target.
4. **Feature engineering:** build term-structure features plus lagged variance and lagged return covariates.
5. **Modeling:** compare observable trailing-variance persistence with ridge regression.
6. **Evaluation:** purge training labels that overlap the forecast horizon, run per-asset walk-forward folds, and aggregate pooled plus per-asset metrics.
7. **Outputs:** write compact artifacts under `outputs/runs/<run_id>/` including notebook subset files.

## Domain Logic

Let $r_t$ be daily log return at date $t$. Then:

$$
rv_t = r_t^2
$$

For forecast horizon $H$ days, forward realized variance is:

$$
RV_{t,t+H} = \sum_{i=1}^{H} rv_{t+i}
$$

Annualized target with annualization factor $A$ is:

$$
RV^{ann}_{t,t+H} = \frac{A}{H} RV_{t,t+H}
$$

Term-structure features include front-second spread, annualized roll yield, and normalized slope.

At a test origin $T$, a training label dated $s$ is usable only when
$s+H\leq T$. The last training row is therefore $T-H$, leaving a purge of
$H-1$ rows relative to an ordinary adjacent train/test split. Persistence uses
the trailing $H$-session annualized variance known at $T$, not a shifted future
label.

## Assumptions And Limits

- Default runtime is offline-only from `data/raw`.
- ClickHouse refresh is optional and isolated to `futures-roll-refresh-raw`.
- Model family is intentionally small (baseline + ridge) for explainability.
- Evaluation uses horizon-purged walk-forward folds per asset to prevent
  training-label overlap with each test period.
- Notebook subset files are compact by design and may not include every full-run row.
