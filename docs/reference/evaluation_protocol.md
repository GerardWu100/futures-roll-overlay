# Evaluation Protocol

The pipeline evaluates forecasts with ordered walk-forward splits per asset.

Protocol:

1. Sort one asset dataset by date.
2. Use configured `train_size`, `test_size`, and `step_size`.
3. For each fold:
   - train model on train window,
   - score on following test window,
   - record `mse`, `rmse`, `mae`, and `r2`.
4. Aggregate fold metrics:
   - per model and per asset,
   - pooled summary across assets.

Models currently evaluated:

- `persistence` baseline (lagged target)
- `ridge` regression

Primary output file:

- `outputs/runs/<run_id>/metrics.csv`
