# Evaluation Guide

This folder defines model evaluation behavior.

- `metrics.py`: regression metrics (`mse`, `rmse`, `mae`, `r2`).
- `walk_forward.py`: ordered train/test split generation and walk-forward ridge evaluation.

Interview story for this layer:

1. Use explicit out-of-sample protocol instead of in-sample fit quality.
2. Preserve temporal ordering to avoid lookahead bias.
3. Return fold-level details plus aggregate metrics for transparency.
