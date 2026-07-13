# Models Guide

This folder contains forecast model logic.

- `baselines.py`: feasible persistence from the trailing variance window known at the forecast date.
- `train.py`: explainable ridge regression fit using closed-form linear algebra.

Interview story for this layer:

1. Baseline first, then modest trained model.
2. Keep coefficient-level interpretability.
3. Avoid model complexity that weakens explanation quality.
