# Outline proposal

## Project scan summary

- Project archetype candidate: mixed `data-pipeline` and `strategy-backtest`, with the second component interpreted as a forecast evaluation rather than a trading backtest.
- Supporting evidence from files: `continuous_futures/build.py` constructs roll-adjusted price histories; `features/term_structure.py` builds curve features; `features/realized_variance.py` defines a forward target; `evaluation/walk_forward.py` enforces temporal train/test ordering; the fresh offline run provides model diagnostics for ES, CL, and GC.

## Blueprint selection

- Selected blueprint: mixed.
- Why this blueprint fits this project: the research result depends as much on the data construction and target timing as on the ridge regression. Treating it as a model-only post would hide the main sources of leakage and measurement error.
- Planned section order:
  1. The practical question: can today's curve help forecast the next two days of variance?
  2. Turning expiring contracts into a usable history.
  3. Auditing the target: how a backward-looking rolling window moved it by one session.
  4. Reading contango, backwardation, and normalized slope.
  5. Walk-forward evaluation, label availability, and the required horizon purge.
  6. Corrected results: neither model wins consistently, and ridge misses the tails.
  7. What the negative result says, and what the experiment cannot establish.

## Planned equations

1. Daily log return and forward realized variance:
   - Purpose: define the forecast target and show that the horizon begins at `t + 1`.
   - Symbols: adjusted close `P_t`, log return `r_t`, horizon `H`, annualization factor `A`, target `RV^{ann}_{t,t+H}`.
   - Delimiter: display.
2. Annualized roll yield:
   - Purpose: convert the front/second contract price difference into a maturity-normalized curve feature.
   - Symbols: front price `F_{1,t}`, second price `F_{2,t}`, calendar-day maturity gap `d_t`, roll yield `y_t`.
   - Delimiter: display.
3. Ridge objective:
   - Purpose: explain coefficient shrinkage and the role of standardized features.
   - Symbols: response `y_i`, standardized feature vector `x_i`, intercept `b`, coefficient vector `beta`, penalty `lambda`, sample count `n`.
   - Delimiter: display.
4. Out-of-sample coefficient of determination:
   - Purpose: interpret the negative reported values against a fold mean forecast.
   - Symbols: observations `y_i`, forecasts `hat y_i`, fold mean `bar y`, test count `m`.
   - Delimiter: display.

## Planned code excerpts

1. File: `src/futures_roll_overlay/features/realized_variance.py`
   - Function/block: explicit lead construction from `t + 1` through `t + H`.
   - Why include this excerpt: it demonstrates the critical exclusion of the current day's return from the target.
2. File: `src/futures_roll_overlay/evaluation/walk_forward.py`
   - Function/block: train/test slice boundaries.
   - Why include this excerpt: it makes the no-lookahead protocol concrete without dumping the entire pipeline.

## Planned technical graphs

1. Graph type: grouped bar chart of per-asset root mean squared error (RMSE).
   - Source (reuse or generate): generate from frozen `metrics.csv` produced by the 2026-07-13 offline run.
   - Expected takeaway: after correcting target alignment and label availability, persistence wins for ES while ridge wins for CL and GC; every mean fold $R^2$ is negative.
2. Graph type: two-panel realized-versus-predicted diagnostic.
   - Source (reuse or generate): generate from frozen `predictions.parquet` from the same run.
   - Expected takeaway: ridge produces negative forecasts and misses large realized-variance observations; persistence is imperfect but better behaved.

## Risks, gaps, and assumptions

- Data gaps: one year of local daily data and only three futures roots; no transaction costs or economic value calculation because this is forecasting research, not a tradable overlay.
- Assumptions: sample calendar rolls, ratio adjustment, corrected two-session target horizon, feasible trailing-variance persistence, a one-row fold purge for label availability, 252 sessions for annualization, an initial test boundary at row 80, 20-row test windows and steps, and ridge penalty `lambda = 1`.
- Validation checks to run before final draft: test the repaired production code; rerun the pipeline; freeze exact metrics and predictions through the production entrypoint; regenerate both graphs; check English/French protected blocks and image references; and run the post validator on both pages.
- Deployment note: canonical files stay under `futures-roll-overlay/blog/`. Per the user's explicit instruction, there is no publish bundle and no copying, building, committing, or pushing in `~/projects/website` during this task.
