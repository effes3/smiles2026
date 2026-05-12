"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that detects hallucinations
from hidden-state features.  Called from ``solution.py`` via ``evaluate.run_evaluation``.
All four public methods (``fit``, ``fit_hyperparameters``, ``predict``,
``predict_proba``) must be implemented and their signatures must not change.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_RANDOM_SEED_ENV = "PROBE_RANDOM_SEED"
_RANDOM_SEED: int = 42

_PCA_COMPONENTS = 128


def _get_random_seed() -> int:
    """Return the probe seed, optionally overridden by an environment variable."""
    raw_seed = os.environ.get(_RANDOM_SEED_ENV)
    if raw_seed is None:
        return _RANDOM_SEED
    try:
        return int(raw_seed)
    except ValueError as exc:
        raise ValueError(f"{_RANDOM_SEED_ENV} must be an integer") from exc


def _seed_everything(seed: int = _RANDOM_SEED) -> None:
    """Seed libraries used by the probe for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pca_n_components(n_features: int, n_samples: int) -> int:
    """Cap PCA rank so sklearn never errors on small folds."""
    return min(_PCA_COMPONENTS, n_features, max(1, n_samples - 1))


def _build_pipeline(n_features: int, n_samples: int, random_state: int) -> Pipeline:
    n_components = _pca_n_components(n_features, n_samples)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(n_components=n_components, random_state=random_state),
            ),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128,),
                    alpha=0.01,
                    early_stopping=True,
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


class HallucinationProbe(nn.Module):
    """Binary classifier: StandardScaler → PCA → ``MLPClassifier``.

    Retains ``torch.nn.Module`` for API compatibility with ``solution.py``;
    inference uses scikit-learn only (no ``forward`` training path).
    """

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Pipeline | None = None
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise RuntimeError(
            "HallucinationProbe uses a sklearn Pipeline; use predict_proba instead."
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        seed = _get_random_seed()
        _seed_everything(seed)
        n_samples, n_features = X.shape
        self._pipeline = _build_pipeline(n_features, n_samples, seed)
        self._pipeline.fit(X, y.astype(np.int64))
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() before fit_hyperparameters().")
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._pipeline.predict_proba(X)
