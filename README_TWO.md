I have analyzed the most recent scientific papers (2024–2026) dedicated to hallucination detection via hidden states (white-box / hidden states probing). Research in this area is currently evolving incredibly fast, and utilizing these approaches will give you a massive advantage in the competition.

Here are the 4 most powerful, scientifically backed ideas from recent publications that fit perfectly into your repository's architecture (specifically in the `aggregation.py` and `probe.py` files).

---

### 1. The "ICR Probe" Method (Dynamics instead of Statics)

**Paper:** *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs* (ACL, July 2025)
**Core idea:** The authors proved that searching for hallucinations solely using "static" hidden states (just taking a single layer) works poorly. When an LLM hallucinates, the process of updating its knowledge in the residual stream (between layers) becomes chaotic. They introduced the **ICR (Information Contribution to Residual Stream)** metric.
**How to apply in `aggregation.py`:**
Instead of just averaging the layers, calculate the difference (delta) between middle or late layers. If the response vector at the 12th layer differs radically from the vector at the 18th layer (e.g., by cosine distance), the model is "hesitating" and is likely fabricating a fact.

```python
# Idea for aggregation.py
layer_A = hidden_states[12] # e.g., 12th layer
layer_B = hidden_states[18] # 18th layer
# Calculate the L2-norm of the difference (how much the state has changed)
layer_diff_norm = np.linalg.norm(layer_B - layer_A, axis=-1).mean()
# You can add this as a separate feature!
```

### 2. The "INSIDE / Eigenscore" Method (Spectral Token Analysis)

**Papers:** *INSIDE: LLMs' internal states retain the power of hallucination detection* (ICLR 2024) and *Hallucination Detection in LLMs Using Spectral Features* (February 2025).
**Core idea:** The authors examined the hidden state matrix for all generated response tokens. They discovered that during a hallucination, the latent space "spreads out". The best way to measure this is to calculate the covariance matrix of the token vectors and extract its **eigenvalues**.
**How to apply in `aggregation.py`:**
Instead of a basic Mean Pooling (averaging across tokens), perform an SVD / PCA on the fly for a single text.

```python
# Idea for aggregation.py (for a single sample)
# response_states has the shape (num_tokens, hidden_dim)
cov_matrix = np.cov(response_states, rowvar=False)
eigenvalues = np.linalg.eigvalsh(cov_matrix)
# Take the top 5 largest eigenvalues
top_eigenvalues = eigenvalues[-5:] 
# This is a super-powerful feature indicating the variance of the model's "thoughts"
```

### 3. "Representation Engineering (RepE)" Method (Truth Direction)

**Papers:** *Taxonomy of Representation Engineering for LLMs* (March 2025) and *SHARP: Steering Hallucination in LVLMs* (November 2025).
**Core idea:** Models contain an internal linear direction (vector) that corresponds to the concept of "truth / falsehood". This vector can be easily calculated using the "Difference-in-Means" method.
**How to apply in `probe.py`:**
During the `fit` (training) stage, you separate the hidden states into two classes: hallucinations (`class=1`) and truth (`class=0`). You calculate the mean vector for class 1 and the mean vector for class 0. Subtract one from the other to get the **"Truth Direction" (or Lie Direction)**. Next, you simply project every new test phrase onto this vector (calculate the dot product). Sometimes, this works better than complex machine learning!

```python
# Idea for probe.py (inside the fit method)
truth_states = X_train[y_train == 0]
hallucination_states = X_train[y_train == 1]

mean_truth = np.mean(truth_states, axis=0)
mean_hallucination = np.mean(hallucination_states, axis=0)

# The hallucination direction vector
self.hallucination_direction = mean_hallucination - mean_truth

# Inside the predict/predict_proba method:
# Project test data onto the vector (dot product)
scores = np.dot(X_test, self.hallucination_direction)
# You can then pass these scores through a sigmoid or calibrate them
```

### 4. "Semantic Entropy Probes (SEP)" / SAPLMA Method

**Papers:** *Semantic Entropy Probes: Robust and Cheap Hallucination Detection* (2024) and *Do LLMs Know about Hallucination?* (February 2024 / March 2025).
**Core idea:** Studies show that the model "realizes" it doesn't know the answer as early as the prompt reading stage. The $s_1$ token (the last token of the prompt) contains more information about an upcoming hallucination than the generated tokens themselves.
**How to apply in `aggregation.py` and `probe.py`:**
1.  Make sure to extract the vector of the **last prompt token (the question)** from the middle layers.
2.  Concatenate it with the mean vector of the response.
3.  Feed this combined representation into a **Linear Logistic Regression** with strict L2 regularization. Papers prove that a linear classifier on hidden states (Linear Probe) generalizes better than non-linear networks, especially when the training set is small.

---

### 🚀 Summary: How to build a State-of-the-Art solution for the competition

If I were implementing this solution to win, my pipeline would look like this:

1.  **In `aggregation.py`:**
    *   Extract `hidden_states` from the 14th to the 20th layer (mid-late layers).
    *   Extract the vector of the last prompt token (to understand if the model knew the answer before generation started).
    *   Extract the generated response tokens and calculate their **Eigenscore** (top-3 eigenvalues of the covariance matrix — a measure of uncertainty).
    *   Calculate the norm of the difference between layers (ICR Score — a metric of dynamics).
    *   Concatenate all of this into a single flat feature vector.
2.  **In `splitting.py`:**
    *   Strictly use `StratifiedKFold(n_splits=5)` to prevent overfitting on 689 samples.
3.  **In `probe.py`:**
    *   Apply `StandardScaler()`.
    *   Apply `PCA(n_components=64)` for noise filtering.
    *   Train a `LogisticRegression(class_weight='balanced')`. Alternatively, use the "Truth Direction" computation (Difference-in-Means) as described in point 3.

In your `SOLUTION.md` report, you can directly cite these papers (e.g., RepE and ICR Probe). This will show the jury (or reviewers) that your solution is not based on random hyperparameter tuning, but rather on the cutting edge of data science.