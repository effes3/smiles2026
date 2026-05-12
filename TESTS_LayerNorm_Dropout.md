# LayerNorm + Dropout MLP + Layer X + Last Token

Probe head:

```python
nn.Sequential(
    nn.LayerNorm(input_dim),
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 1),
)
```

Each row uses `LAST_TOKEN_LAYER_INDEX=X` and runs `python solution.py` with `PROBE_RANDOM_SEED` set to `42`, `43`, and `44`.

`Std %` is sample standard deviation in percentage points. `Std raw` is sample std on the 0..1 AUROC scale.

| X | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 71.12% | 72.89% | 72.56% | 72.19% | 0.94 | 0.009396 |
| 22 | 74.15% | 73.40% | 73.95% | 73.83% | 0.39 | 0.003892 |
| 23 | 72.38% | 71.85% | 71.54% | 71.93% | 0.42 | 0.004246 |
