# K-Fold Results: Geometry + AdamW Scheduler + Layer X

Configuration:

- `splitting.py`: 5-fold stratified split with validation slice inside each fold.
- `aggregation.py`: selected layer last-token vector plus three geometric scalars: L2 norm, first/last cosine similarity, first/last Euclidean distance.
- `probe.py`: ReLU MLP (`Linear -> ReLU -> Linear`) trained with AdamW and cosine scheduler.
- Each cell is `avg_test_auroc` from `results.json`, averaged across the 5 folds produced by `solution.py`.

`Std %` is sample standard deviation across seeds in percentage points. `Std raw` is sample std on the 0..1 AUROC scale.

| X | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw | Mean Acc | Mean F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | 73.32% | 72.83% | 73.33% | 73.16% | 0.29 | 0.002862 | 71.03% | 82.02% |
| 22 | 73.28% | 73.12% | 72.84% | 73.08% | 0.22 | 0.002206 | 71.56% | 82.13% |
| 23 | 73.56% | 73.82% | 73.66% | 73.68% | 0.13 | 0.001334 | 70.73% | 81.55% |
