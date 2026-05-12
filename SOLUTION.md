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

2. **Cheap geometry beats blind magnitude**: replacing raw **‖h23‖** with **‖h23‖/‖h22‖** and adding **first–last** cosine and distance on L23 gave the **best K-fold mean test AUROC** we archived (**~73.77%**, `TESTS_Ratio.md`).

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

### Exact command to reproduce `predictions.csv` and `results.json`

From the repository root:

```bash
python solution.py
```

Optional: fix the probe RNG for repeatable training:

```bash
export PROBE_RANDOM_SEED=42   # Linux / macOS
python solution.py
```

Artifacts written by the official script (paths are asserted in `solution.py`):

- `results.json` — K-fold summary and per-fold metrics from `evaluate.py`
- `predictions.csv` — labels for `data/test.csv` from the probe fit on all non-test training indices implied by the split list

### Important implementation details

- **Frozen model**: `model.py` loads **Qwen/Qwen2.5-0.5B** with `output_hidden_states=True`; do not rely on editing `model.py` or `evaluate.py` for the competition.
- **Features**: built in `aggregation.py` from `torch.stack(outputs.hidden_states, …)` per sample; shape depends only on that file.
- **Splits**: `splitting.py` returns **five** disjoint outer folds; `evaluate.py` trains a fresh probe per fold and averages metrics.

---

## Part 3 — Final solution (what we ship in this repo)

### Components modified

| File | Role |
|------|------|
| `aggregation.py` | Maps `(hidden_states, attention_mask)` → **1-D feature vector** for each example. |
| `probe.py` | **`HallucinationProbe`**: `StandardScaler` + small MLP + threshold tuning API. |
| `splitting.py` | **`split_data`**: stratified **5-fold** outer CV + stratified **validation** slice per fold. |

### Current feature design (`aggregation.py`)

- **Last-token readouts** at hidden-state indices **19, 22, 23** (896 dims each → 2688 dims).
- **Spectral-style scalars** (Noël 2026, Sec. 3.3-style definitions) computed at **L23** (entropy), **L17** and **L22** (HFER), **L19** (smoothness), using a **chain graph** on real tokens: edge weight `relu(cos(h_i, h_{i+1})) + ε`, then combinatorial Laplacian **L = D − W**, eigenbasis **GFT**, energy split for **HFER**, Dirichlet quotient for **smoothness**, Shannon-like energy entropy for **spectral entropy**.
- **Norm ratio** **‖h23_last‖₂ / ‖h22_last‖₂** (1 dim).

**Total feature dimension:** **2693** = 3×896 + 4 + 1.

### Current probe (`probe.py`)

- `StandardScaler` on **numpy** features.
- MLP: **`Linear(d,256) → SiLU → Dropout(0.4) → Linear(256,1)`**.
- **200** epochs, **AdamW** (`lr=1e-3`, `weight_decay=1e-2`), **CosineAnnealingLR** (`T_max=200`).
- **BCEWithLogitsLoss** with **`pos_weight = n_neg / n_pos`** for class imbalance.
- **`fit_hyperparameters`**: F1-optimal probability threshold on the **validation** indices of each fold.

### What helped the metric most (evidence-based)

Across archived K-fold + multi-seed logs, the largest gain came from **(a)** stratified **K-fold** evaluation for honest model selection, **(b)** **late-layer** readouts with **(c)** **norm ratio and simple geometry** at L23 (`TESTS_Ratio.md`). The **current** repository state documents a **spectral proxy** extension (`TESTS_FINAL.md`); its K-fold mean test AUROC was **~72.6%** in the logged run — informative, but **not** the best number on the labelled protocol compared to the **L23 ratio geometry** line.

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

We treated **internal test AUROC** (5-fold mean on `dataset.csv`) as the compass, while remembering the **official** score is on **hidden `test.csv`**. The story above is the audit trail of how that compass moved from a **single-split baseline** to **geometry-aware late layers** to **K-fold realism**—and why the **simplest strong recipe** (L23 + ratio + geometry + PyTorch MLP) remains the **best documented** labelled-set configuration even as the codebase explores richer signals.
