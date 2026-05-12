# K-Fold Results: Layers 6 & 22 + Norm Ratio + Cosine

Configuration:

- `splitting.py`: 5-fold stratified split with validation slice inside each fold.
- `aggregation.py`: last-token `h6_last` and `h22_last`; scalar `||h22_last||_2 / ||h21_last||_2`; cosine similarity between `h6_last` and `h22_last` at the last real token.
- Feature dimension: `896 + 896 + 1 + 1 = 1794`.
- `probe.py`: `StandardScaler` then `Linear(input_dim, 256) -> SiLU -> Dropout(0.4) -> Linear(256, 1)`; `AdamW(lr=1e-3, weight_decay=1e-2)`; `CosineAnnealingLR` over 200 steps; `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos`; threshold tuning via `fit_hyperparameters` on the validation split.
- Each AUROC cell is `avg_test_auroc` from `results.json`, averaged across the 5 folds produced by `solution.py`.

Runs: `python solution.py` with `PROBE_RANDOM_SEED` set to `42`, `43`, and `44`.

| Feature set | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw | Mean Acc | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h6 + h22 + norm22/norm21 + cos(h6,h22) + SiLU MLP | 73.37% | 72.97% | 73.17% | 73.17% | 0.20 | 0.00201 | 73.06% | 83.09% |

Raw `avg_test_auroc` values: `0.733726`, `0.729713`, `0.731665` (sample standard deviation on the 0–1 scale ≈ `0.00201`).
