I have reviewed the structure and description of the **SMILES-2026-Hallucination-Detection** repository.

### 🔍 Task Analysis and Main Challenge

**Core idea:** Build a classifier (probe) to detect hallucinations in the responses of the Qwen2.5-0.5B LLM using its internal hidden states.
**Main problem:** The training dataset size is only **689 labeled examples**. At the same time, the dimensionality of the hidden states is massive (24 layers $\times$ 896 features $\times$ number of tokens). Under these conditions, complex models will be prone to catastrophic **overfitting**.
**Key to success:** Strict overfitting control, smart feature engineering based on LLM knowledge, and simple, robust classifiers. You are only allowed to modify 3 files: `splitting.py`, `aggregation.py`, and `probe.py`.

Below is a concept for a high-quality solution, broken down by file.

---

### 1. Validation Strategy (`splitting.py`)
A standard random train/test split on 689 examples will yield a highly unstable evaluation. You might accidentally get an "easy" or "hard" split, which will distort the Accuracy metric.
*   **Idea:** Use **Stratified K-Fold** (with 5 or 10 folds) from `sklearn.model_selection`. Stratification is mandatory to preserve the original class balance (hallucination / truth) in each fold.
*   **Ensembling:** To generate final predictions for `test.csv` (where labels are hidden), run the features through **all K models** trained on the folds and average their predictions (probabilities). This will provide a significant boost to stability and the final metric on the hidden test set.

---

### 2. Feature Extraction (`aggregation.py`)
This is where the main magic happens. In the competition description, the authors directly advise experimenting with geometric and topological methods.

*   **Layer Selection:**
    *   Do not use the final layers. In LLMs, they are heavily "tuned" for the next-token prediction task (vocab projection) rather than factuality.
    *   **Optimal:** Middle and mid-late layers. For a 24-layer model, these are typically layers **12 through 20**. You can extract and concatenate them.
*   **Token Pooling:**
    The matrix has a variable length due to the different number of generated tokens. You need to obtain fixed-length vectors:
    1.  `Mean Pooling` — the average vector across all tokens of the *generated response*.
    2.  `Max Pooling` — the maximum values across the response tokens (helps to "catch" sharp activation spikes on specific hallucinated words).
    3.  `Last Prompt Token` — the vector of the final token of the prompt itself (contains the compressed context of the task).
*   **Geometric Features:**
    Add new scalar values to your vectors, which, according to scientific papers (e.g., on Representation Engineering), correlate with model uncertainty:
    *   **L2-norm ($||x||_2$):** Often, when the model hallucinates and is "uncertain," the norm of its hidden state vectors drops.
    *   **Cosine Similarity:** Calculate the distance between the last prompt token and the averaged response vector. A strong deviation may indicate that the model is drifting off-topic (hallucination drift).
    *   **Trajectory Length:** The sum of Euclidean distances between adjacent generation tokens ($t_1 \to t_2 \to t_3$). During hallucinations, the generation process in the latent space becomes more chaotic.

---

### 3. Classifier and Dimensionality Reduction (`probe.py`)
The organizers hint: *"you may apply dimensionality reduction techniques within probe.py"*. This is a direct guideline on how to deal with the small amount of data.

*   **Normalization (important!):** Before training, be sure to run the features through a `StandardScaler` (Z-score normalization). Without it, dimensionality reduction algorithms and linear models will not work correctly.
*   **Dimensionality Reduction:**
    The gathered features (thousands of dimensions) need to be compressed down to **32–128 principal components** using **PCA (Principal Component Analysis)**. This will act as a noise filter and save you from the curse of dimensionality. An alternative is UMAP, but PCA is more stable on small datasets.
*   **The Probe Model (HallucinationProbe):**
    Forget about complex multi-layer neural networks (MLPs); on 689 samples, they will instantly memorize the data and fail on the test set. Use simple, easily regularizable algorithms:
    1.  **Logistic Regression** with strong L2 regularization (`C=0.1` or `0.05`).
    2.  **Linear SVM** (`LinearSVC`).

    Ultimately, the logic of your `probe.py` can be elegantly wrapped into a pipeline:
    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    # Inside your HallucinationProbe class:
    self.model = Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=64)), 
        ('clf', LogisticRegression(C=0.1, class_weight='balanced'))
    ])
    ```

---

### 🏆 What to include in `SOLUTION.md` (Report)
According to the competition rules, you must describe unsuccessful experiments. To make your report look like the work of someone who deeply immersed themselves in the topic, you can implement and honestly discard the following approaches (they will almost certainly yield worse results):
1.  *Using all 24 layers of Qwen2.5-0.5B:* Led to excessively high dimensionality and overfitting. Selecting the middle layers provided a more stable signal for factuality.
2.  *Using neural networks (MLP / Deep Layers):* The neural network easily achieved ~100% accuracy on the train set but completely failed on validation due to a critical lack of training data, even with Dropout. A simple logistic regression proved to be more robust.
3.  *Analyzing only the prompt tokens:* The model could not distinguish a hallucination from the truth because, for the exact same input prompt, the LLM can generate both a correct and a fabricated response.

**Summary:** Your success in this competition depends not on the complexity of the classifier, but on how informative the features you engineer in `aggregation.py` are (especially the geometric properties) and how strictly you prevent the model from overfitting in `probe.py`. Good luck!