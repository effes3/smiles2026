# K-Fold Results: Dual-Core Layers 12 & 23 + Norm Ratio + PCA–MLP Pipeline

Configuration:

- `splitting.py`: 5-fold stratified split with validation slice inside each fold (`random_state=42` in `split_data`).
- `aggregation.py`: last-token `h12`, last-token `h23`, and scalar `||h23_last||_2 / ||h22_last||_2` → feature dimension `896 × 2 + 1 = 1793`.
- `probe.py`: `Pipeline([StandardScaler(), PCA(n_components=min(128, n_features, n_train_samples − 1)), MLPClassifier(hidden_layer_sizes=(128,), alpha=0.01, early_stopping=True, max_iter=1000)])` with `random_state` from `PROBE_RANDOM_SEED`. On these folds, `n_train ≈ 447`, so PCA uses **128** components.
- Each AUROC cell is `avg_test_auroc` from `results.json`, averaged across the 5 folds produced by `solution.py`.

Runs: `python solution.py` with `PROBE_RANDOM_SEED` set to `42`, `43`, and `44`.

| Feature set | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw | Mean Acc | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| concat(h12_last, h23_last, norm23/norm22) + Scaler→PCA(128)→MLP(128) | 62.83% | 66.32% | 69.45% | 66.20% | 3.31 | 0.0331 | 70.25% | 81.84% |

Raw `avg_test_auroc` values: `0.6283`, `0.6632`, `0.6945` (sample standard deviation on the 0–1 scale ≈ `0.0331`).
