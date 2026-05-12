# K-Fold Results: Layer 23 Delta Features + Geometry

Configuration:

- `splitting.py`: 5-fold stratified split with validation slice inside each fold.
- `aggregation.py`: `h23_last`, `h23_last - h22_last`, `h23_last - h21_last`, plus geometry scalars from layer `23`.
- Feature dimension: `3 * 896 + 3 = 2691`.
- Geometry scalars: `||last_state||_2`, first/last cosine similarity, first/last Euclidean distance.
- `probe.py`: ReLU MLP (`Linear -> ReLU -> Linear`) trained with AdamW and cosine scheduler.
- Each AUROC cell is `avg_test_auroc` from `results.json`, averaged across the 5 folds produced by `solution.py`.

| Feature Set | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw | Mean Acc | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h23 + (h23-h22) + (h23-h21) | 73.52% | 73.18% | 72.86% | 73.19% | 0.33 | 0.003314 | 70.88% | 81.41% |
