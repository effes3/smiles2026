# K-Fold Results: Spectral-Style Features (L19 / L22 / L23) + PyTorch MLP

Configuration:

- `splitting.py`: 5-fold stratified split with validation slice inside each fold (`random_state=42` in `split_data`).
- `aggregation.py` (spectral guardrail–inspired, Noël 2026 arXiv:2602.08082, Sec. 3.3 formulas): last-token `h19_last`, `h22_last`, `h23_last`; spectral **entropy** at layer **23**, **HFER** at layers **17** and **22**, **smoothness** at layer **19**; scalar **‖h23_last‖₂ / ‖h22_last‖₂**. Because `solution.py` does not return attention weights, the Laplacian uses a **path graph** on real tokens with edge weights `relu(cos(h_i, h_{i+1})) + ε` at each layer (same SE / HFER / smoothness definitions on that proxy graph).
- Feature dimension: `896 × 3 + 4 + 1 = 2693`.
- `probe.py`: `StandardScaler` then `Linear → SiLU → Dropout(0.4) → Linear`; AdamW (`lr=1e-3`, `weight_decay=1e-2`); `CosineAnnealingLR` over 200 epochs; `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos`; validation F1 threshold tuning via `fit_hyperparameters`.
- Each AUROC cell is `avg_test_auroc` from `results.json`, averaged across the 5 folds from `solution.py`.

Runs: `python solution.py` with `PROBE_RANDOM_SEED` ∈ `{42, 43, 44}` (extraction ~25–26 s per run on this machine).

| Feature set | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC | Std % | Std raw | Mean Acc | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L19/L22/L23 + chain-graph SE/HFER/S + norm23/22 + SiLU MLP | 72.64% | 72.65% | 72.41% | 72.57% | 0.14 | 0.00138 | 70.54% | 81.36% |

Raw `avg_test_auroc`: `0.726430`, `0.726491`, `0.724068` (sample standard deviation on the 0–1 scale ≈ `0.00138`).
