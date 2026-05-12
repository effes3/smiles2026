# Geometry Features + ReLU MLP + Layer X + Last Token

Aggregation returns selected layer last-token vector plus three scalar geometric features:

- `||last_state||_2` (vector magnitude)
- `cosine(first_state, last_state)` (semantic drift proxy; higher = more aligned)
- `||last_state - first_state||_2` (relative distance)

Important: `aggregation.py` does not receive the true prompt/response boundary, so `first_state` vs `last_state` is a no-`solution.py` proxy for prompt-to-response drift.

Probe head is the original control ReLU MLP: `Linear(input_dim, 256) -> ReLU -> Linear(256, 1)`.

Each row uses `LAST_TOKEN_LAYER_INDEX=X` and runs `python solution.py` with `PROBE_RANDOM_SEED` set to `42`, `43`, and `44`.

`Std %` is sample standard deviation in percentage points. `Std raw` is sample std on the 0..1 AUROC scale.

| X | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 75.59% | 74.86% | 73.57% | 74.67% | 1.02 | 0.010179 |
| 22 | 75.81% | 75.78% | 76.09% | 75.89% | 0.17 | 0.001726 |
| 23 | 72.80% | 73.13% | 73.57% | 73.17% | 0.39 | 0.003880 |
