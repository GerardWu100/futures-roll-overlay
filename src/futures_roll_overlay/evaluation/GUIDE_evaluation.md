# Evaluation Guide

This folder defines model evaluation behavior.

- `metrics.py`: regression metrics (`mse`, `rmse`, `mae`, `r2`).
- `walk_forward.py`: horizon-purged train/test split generation and walk-forward ridge evaluation.

Interview story for this layer:

1. Use explicit out-of-sample protocol instead of in-sample fit quality.
2. End training at $T-H$ so every forward label is observable at the first test date $T$.
3. Return fold-level details plus aggregate metrics for transparency.
