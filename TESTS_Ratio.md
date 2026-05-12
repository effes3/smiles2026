# K-Fold Results: Layer 23 Norm Ratio Geometry

Configuration:

- `splitting.py`: 5-fold stratified split with validation slice inside each fold.
- `aggregation.py`: `h23_last` plus geometry scalars from layer `23`.
- Replaced raw `||h23_last||_2` with `||h23_last||_2 / ||h22_last||_2`.
- Other geometry scalars: first/last cosine similarity and first/last Euclidean distance at layer 23.
- Feature dimension: `896 + 3 = 899`.
- `probe.py`: ReLU MLP (`Linear -> ReLU -> Linear`) trained with AdamW and cosine scheduler.
- Each AUROC cell is `avg_test_auroc` from `results.json`, averaged across the 5 folds produced by `solution.py`.

| Feature Set | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw | Mean Acc | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h23 + norm23/norm22 + cosine + distance | 73.68% | 73.91% | 73.73% | 73.77% | 0.12 | 0.001186 | 70.68% | 81.59% |
