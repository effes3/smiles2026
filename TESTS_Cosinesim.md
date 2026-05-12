# K-Fold Results: Layer 23/22 Cosine Similarity Geometry

Configuration:

- `splitting.py`: 5-fold stratified split with validation slice inside each fold.
- `aggregation.py`: `h23_last` plus geometry scalars.
- Geometry scalars: `||h23_last||_2 / ||h22_last||_2`, `cosine(h23_last, h22_last)`, and first/last Euclidean distance at layer 23.
- Feature dimension: `896 + 3 = 899`.
- `probe.py`: ReLU MLP (`Linear -> ReLU -> Linear`) trained with AdamW and cosine scheduler.
- Each AUROC cell is `avg_test_auroc` from `results.json`, averaged across the 5 folds produced by `solution.py`.

| Feature Set | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw | Mean Acc | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h23 + norm23/norm22 + cosine23_22 + distance | 73.50% | 73.74% | 73.84% | 73.70% | 0.17 | 0.001740 | 70.78% | 81.61% |
