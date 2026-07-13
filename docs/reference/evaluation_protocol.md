# Evaluation Protocol

The pipeline evaluates forecasts with ordered walk-forward splits per asset.

Protocol:

1. Sort one asset dataset by date.
2. Use configured `train_size`, `test_size`, and `step_size`; `train_size`
   locates the first test row rather than promising that every earlier label is
   observable.
3. For each fold:
   - let $T$ be the first test date and $H$ the forward-label horizon,
   - end model training at $T-H$, which purges the final $H-1$ nominal rows,
   - train the model only on labels observable at $T$,
   - score on following test window,
   - record `mse`, `rmse`, `mae`, and `r2`.
4. Aggregate fold metrics:
   - per model and per asset,
   - pooled summary across assets.

Models currently evaluated:

- `persistence` baseline (trailing $H$-session variance known at forecast time)
- `ridge` regression

Primary output file:

- `outputs/runs/<run_id>/metrics.csv`
