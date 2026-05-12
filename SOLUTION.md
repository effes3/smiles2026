# SMILES-2026 — Solution report

This document satisfies the competition report requirements: **reproducibility**, **final solution**, and **experiments / failed attempts**. It opens with a **short story** of how the work evolved, ordered by the **first git commit** that introduced each experiment log (`*.md`).

---

## Part 1 — Story: what we did (timeline) and what we learned

### Timeline from git (first appearance of each log)

| When | What was added | Intent |
|---------------------------|------------------|--------|
| **2026-03-25** | `README.md` (initial import) | Project scaffold. |
| **2026-05-06** | `Cloned it` | Exploring it in Google Colab. |
| **2026-05-12 19:52** | `BASELINE.md`, `README_ONE.md`, `README_TWO.md` | Freeze **single stratified split** metrics with **default student stubs** (896-D last token) so we had a floor to beat. |
| **2026-05-12 20:22** | `TESTS.md` | **Layer sweep**: last-token embedding from each layer × three probe seeds — found late layers (21–24) strongest on that protocol. |
| **2026-05-12 21:18** | `TESTS_SwiGLU.md` | Try a **SwiGLU-shaped** probe head (mirrors Qwen FFN style); logged underperformance vs simple MLP. |
| **2026-05-12 21:22** | `TESTS_LayerNorm_Dropout.md` | Try **LayerNorm + Dropout** in the probe; did not win. |
| **2026-05-12 21:29** | `TESTS_Geometry.md` | Add **geometry scalars** (norm, first–last cosine, distance) at chosen layers; **~75.9%** mean test AUROC for layer 22 on the logged protocol. |
| **2026-05-12 21:44** | `TESTS_KFold.md` | Switch evaluation to **5-fold stratified CV** + inner val for threshold tuning; geometry at layers 21–23. |
| **2026-05-12 21:48** | `TESTS_Stack.md` | **Stack** last tokens from layers 21+22+23; extra capacity did not beat the best compact recipe. |
| **2026-05-12 21:51** | `TESTS_Delta.md` | **Delta** features between layers; still no clear win over the best L23-style geometry. |
| **2026-05-12 21:59** | `TESTS_Ratio.md` | Replace raw norm with **‖h23‖/‖h22‖** ratio + other L23 geometry — **best logged mean test AUROC (~73.77%)** under K-fold + three seeds. |
| **2026-05-12 22:26** | `TESTS_12n23_layer.md`, `TESTS_Cosinesim.md` | **Dual-layer** (12+23) + PCA–sklearn pipeline, and cosine variants; dual-core + PCA **underperformed** PyTorch MLP on the same labelled protocol. |
| **2026-05-12 22:35** | `TESTS_6n22_layer.md` | **Early + late** representation (L6 + L22 + norm + cosine) with SiLU MLP; strong but slightly below the L23 ratio row. |
| **2026-05-12 22:56** | `TESTS_FINAL.md`, `README_orig.md` | **Spectral-style** features (entropy / HFER / smoothness on a **path-graph proxy** + L19/L22/L23 last tokens + norm ratio), inspired by Noël (2026); documented K-fold metrics. |

The dense burst on **12 May 2026** is one focused iteration day: baseline → sweeps → probe variants → geometry → **K-fold honesty** → richer features → spectral story.

### What we learned (facts tied to those runs)

1. **Late layers carry the signal** for this task on Qwen2.5-0.5B: layer sweeps consistently pointed to **L21–L23** (and neighbours), not early layers alone.

2. **Cheap geometry beats blind magnitude**: replacing raw **‖h23‖** with **‖h23‖/‖h22‖** and adding **L23–L22 last cosine** plus **L23 first–last distance** gave the **best K-fold mean test AUROC** we archived (**~73.77%**, `tests/TESTS_Ratio.md`).

3. **K-fold changed the headline number**: after **5-fold** stratified evaluation, **most** logged mean test AUROCs sit **~72–73%**, i.e. **below** many **~75%+** single-split / early-sweep numbers. That is not “the model got worse” — the **metric got more honest** (every point is test once; less variance from one lucky partition).

4. **Bigger vectors are not free**: stacking layers, deltas, dual cores, or spectral stacks increased **dimension and compute** but **did not reliably beat** the compact L23 geometry line on the same evaluation contract.

5. **Probe inductive bias**: **PyTorch MLP** + `StandardScaler` + **class-weighted BCE** + **AdamW** + **cosine LR** + **val F1 threshold** remained the workhorse; SwiGLU-style heads and sklearn **PCA + MLPClassifier** were tried and discarded for this dataset size.

6. **Spectral paper vs our pipeline**: attention–Laplacian features from Noël (2026) need **attention matrices**; **`solution.py` is fixed** and does not pass them, so we implemented the **same diagnostic formulas** on a **hidden-state path graph** (adjacent-token cosine weights). That is a **principled proxy**, not a faithful reproduction of attention topology.

---

## Part 2 — Reproducibility

### Environment

- **Python**: 3.10+ recommended (matches typical `torch` / `transformers` stacks).
- **Hardware**: **CUDA** GPU recommended; CPU runs but are slower. Apple **MPS** is supported by `solution.py` if available.
- **Dependencies**: install from the repository root:

```bash
pip install -r requirements.txt
```

### What “reproducing our results” means

The headline **~73.77% mean test AUROC** on the **labelled** `dataset.csv` is the **average of three** full runs of `python solution.py`, each with a different **`PROBE_RANDOM_SEED`**, as recorded in **`tests/TESTS_Ratio.md`**. Each run overwrites **`results.json`**; copy or rename it between runs if you want to keep all three JSON files.

The **committed** `aggregation.py`, `probe.py`, and `splitting.py` match that log (**899-D** features, **ReLU** MLP, **5-fold** stratified splits).

### Commands to match the published internal AUROC (three seeds)

From the repository root (Linux / macOS):

```bash
export PROBE_RANDOM_SEED=42
python solution.py
# Read avg_test_auroc in results.json or the printed summary.

export PROBE_RANDOM_SEED=43
python solution.py

export PROBE_RANDOM_SEED=44
python solution.py
```

Take the **mean** of the three **`avg_test_auroc`** values. On the machine where the table was built, the per-seed means were about **73.68%**, **73.91%**, and **73.73%** (overall mean **~73.77%**). **Small drift** is normal (GPU nondeterminism, library versions).

**Windows (PowerShell):** `$env:PROBE_RANDOM_SEED=42; python solution.py`

### Single run (default seed, predictions file)

```bash
python solution.py
```

If **`PROBE_RANDOM_SEED`** is unset, the probe uses seed **42** (see `probe.py`). You still get valid **`predictions.csv`** and one **`results.json`**, but only **one** of the three seeds used in the published mean.

### Artifacts

- **`results.json`** — per-fold metrics and summary averages from **`evaluate.py`**
- **`predictions.csv`** — predicted labels for **`data/test.csv`**

### Important implementation details

- **Frozen stack**: do not rely on editing **`model.py`** or **`evaluate.py`** for the competition.
- **Feature dimension**: with the Ratio recipe, the printed line should show **`feature_dim = 899`**.
- **Splits**: **`splitting.py`** returns **five** outer folds; **`evaluate.py`** trains a fresh probe per fold and averages metrics.

---

## Part 3 — Final solution (what we ship in this repo)

### Components modified

| File | Role |
|------|------|
| `aggregation.py` | Maps `(hidden_states, attention_mask)` → **1-D feature vector** for each example. |
| `probe.py` | **`HallucinationProbe`**: `StandardScaler` + **ReLU MLP** + threshold tuning API. |
| `splitting.py` | **`split_data`**: stratified **5-fold** outer CV + stratified **validation** slice per fold. |

### Feature design (`aggregation.py`) — **`tests/TESTS_Ratio.md`**

- **`h23_last`** — last real token at hidden-state index **23** (896 dims).
- **`‖h23_last‖₂ / ‖h22_last‖₂`** — norm ratio (scalar).
- **Cosine** between **`h23_last`** and **`h22_last`** (scalar).
- **Euclidean** **`‖h23_last − h23_first‖`** on layer **23** (scalar).

**Total feature dimension:** **899** = 896 + 3.

### Probe (`probe.py`)

- **`StandardScaler`** on NumPy features.
- MLP: **`Linear(d, 256) → ReLU → Linear(256, 1)`**.
- **200** epochs, **AdamW** (`lr=1e-3`, `weight_decay=1e-2`), **CosineAnnealingLR** (`T_max=200`).
- **`BCEWithLogitsLoss`** with **`pos_weight = n_neg / n_pos`**.
- **`fit_hyperparameters`**: F1-optimal probability threshold on the **validation** indices of each fold.

### What helped the metric most (evidence-based)

Stratified **5-fold** evaluation, **late layer 23**, **norm ratio to layer 22**, **light geometry** (last-token cosine across 23/22 and first–last distance on 23), and this **ReLU MLP** probe — together **~73.77%** mean test AUROC averaged over seeds **42–44** in **`tests/TESTS_Ratio.md`**. Richer experiments (spectral stacks, dual cores, PCA pipelines, etc.) are documented in other **`tests/TESTS_*.md`** files but did **not** beat that line on the same contract.

---

## Part 4 — Experiments and failed (or weaker) attempts

| Idea | Outcome | Where logged |
|------|---------|----------------|
| SwiGLU probe head | Weaker than control MLP | `TESTS_SwiGLU.md` |
| LayerNorm + Dropout probe | Weaker | `TESTS_LayerNorm_Dropout.md` |
| 21+22+23 stack | No beat over best geometry | `TESTS_Stack.md` |
| Inter-layer deltas | No beat | `TESTS_Delta.md` |
| L12+L23 + sklearn **PCA(128)** + `MLPClassifier` | Large drop vs PyTorch MLP | `TESTS_12n23_layer.md` |
| Dual-core L6+L22 + SiLU MLP | Strong, slightly below L23 ratio row | `TESTS_6n22_layer.md` |
| Spectral proxy + multi-layer readouts | Honest topology substitute; K-fold AUROC mid-72% in log | `TESTS_FINAL.md` |

**Why discards happened:** either **no lift** on the stratified K-fold metric vs simpler features, or **worse generalisation** (higher train AUROC with flat or lower test AUROC), or **instability / cost** (PCA pipeline, extra dimensions) without payoff on **N ≈ 689**.

---

## Closing remark

We treated **internal test AUROC** (5-fold mean on `dataset.csv`) as the compass, while remembering the **official** score is on **hidden `test.csv`**. The story above is the audit trail of how that compass moved from a **single-split baseline** to **geometry-aware late layers** to **K-fold realism**. The **submitted code path** matches the best archived K-fold recipe (**`tests/TESTS_Ratio.md`**, **~73.77%** mean over seeds 42–44); see **Part 2** to reproduce those numbers and **Part 3** for the exact feature and probe definitions.

**AUROC peaks (logged):** the **highest before K-fold** is in **`tests/TESTS_Geometry.md`**: mean test AUROC up to **75.89%** (layer 22 with geometry scalars). The **75.45%** value is the **mean** test AUROC for **layer 22** in the **last-token-only** sweep **`tests/TESTS.md`** (no extra geometry). The **highest after K-fold** (5-fold mean over seeds 42–44) is **`tests/TESTS_Ratio.md`**: **73.77%** mean test AUROC (L23 + norm ratio + cosine + distance).
