# K-Fold Results: 21+22+23 Stack + Geometry

Configuration:

- `splitting.py`: 5-fold stratified split with validation slice inside each fold.
- `aggregation.py`: concatenates last-token vectors from layers `21`, `22`, and `23`, then appends geometry scalars from layer `23`.
- Feature dimension: `3 * 896 + 3 = 2691`.
- Geometry scalars: `||last_state||_2`, first/last cosine similarity, first/last Euclidean distance.
- `probe.py`: ReLU MLP (`Linear -> ReLU -> Linear`) trained with AdamW and cosine scheduler.
- Each AUROC cell is `avg_test_auroc` from `results.json`, averaged across the 5 folds produced by `solution.py`.

| Feature Stack | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw | Mean Acc | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 21+22+23 | 73.69% | 73.04% | 73.02% | 73.25% | 0.38 | 0.003822 | 70.93% | 81.72% |
