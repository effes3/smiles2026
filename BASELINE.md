# Baseline metrics (`python solution.py`)

Captured after running the **unmodified** student stubs (`splitting.py`, `aggregation.py`, `probe.py`) with the repo defaults. **Date:** 2026-05-12. **Device:** CUDA (run on this machine).

## Reproduce

```bash
cd /home/semakin_grisha/SMILES-2026-Hallucination-Detection
./.venv/bin/python solution.py
```

Artifacts: `results.json` (full floats), `predictions.csv` (100 rows from `data/test.csv`).

## Defaults (from `solution.py`)

| Setting | Value |
|--------|--------|
| Data | `./data/dataset.csv` |
| Test file | `./data/test.csv` |
| `BATCH_SIZE` | 4 |
| `USE_GEOMETRIC` | `False` |
| Model | `Qwen/Qwen2.5-0.5B` (`model.py`), `MAX_LENGTH` 512 |

## Data snapshot (printed by script)

- **689** labeled rows: **483** hallucinated (1), **206** truthful (0).
- **Feature matrix:** `(689, 896)` — last layer, last non-padding token only (`aggregation.py` default).

## Split (`splitting.py` default)

Single stratified split: **train 481 / val 104 / test 104** (15% test, 15% val of remainder; `random_state=42`).

## Metrics (from `results.json`)

Percentages rounded to two decimals; raw scores in parentheses match `results.json`.

| Checkpoint | Accuracy | F1 | AUROC |
|------------|----------|-----|-------|
| 1. Majority-class baseline (on eval **test** split) | 70.19% (0.7019) | 82.49% (0.8249) | N/A |
| 2. `HallucinationProbe` — **train** | 70.06% (0.7006) | 82.40% (0.8240) | ~100.00% (0.99998) |
| 3. `HallucinationProbe` — **val** | 70.19% (0.7019) | 82.49% (0.8249) | 67.90% (0.6790) |
| 4. `HallucinationProbe` — **test** | 70.19% (0.7019) | 82.49% (0.8249) | **75.39% (0.7539)** |

**Primary metric (per `evaluate.py` summary):** test split **AUROC ≈ 0.754**.

### Averages (`results.json` top-level)

Same as single-fold row above: `avg_test_auroc` **0.7539**, `avg_baseline_accuracy` **0.7019**, `avg_train_auroc` **0.99998**.

## Timing

- Hidden-state extraction (689 samples): **~6.1 s** (`extract_time_s` in `results.json`).

## Console notes (informational)

The run may print Hugging Face hub warnings (unauthenticated requests), a `torch_dtype` deprecation notice from `transformers`, and a verbosity note about `output_hidden_states` generation flags; these do not change the saved metrics.

## How to read the baseline

- **Baseline accuracy/F1** track the **majority class** (hallucinated is more frequent), so accuracy ~70% and high F1 on the positive class are expected for a constant predictor.
- The default **probe** gets **near-perfect train AUROC** but modest **val/test AUROC**, which is consistent with a small MLP overfitting ranking on the training split while threshold tuning collapses discrete predictions toward the majority class on val/test (Accuracy/F1 align with baseline).

Use **test AUROC** as the main number to beat when you change aggregation, splitting, or the probe.
