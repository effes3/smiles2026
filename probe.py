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
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


_RANDOM_SEED_ENV = "PROBE_RANDOM_SEED"
_RANDOM_SEED: int = 42
_FIT_EPOCHS: int = 200


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


class HallucinationProbe(nn.Module):
    """Binary probe: ``StandardScaler`` + Linear(→256) → SiLU → Dropout → Linear(→1)."""

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Module | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits of shape ``(n_samples,)``."""
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train with AdamW, cosine LR schedule, and class-weighted BCE."""
        _seed_everything(_get_random_seed())
        X_scaled = self._scaler.fit_transform(X)

        self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=_FIT_EPOCHS
        )

        self.train()
        for _ in range(_FIT_EPOCHS):
            optimizer.zero_grad()
            logits = self(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
            scheduler.step()

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on validation F1."""
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
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        self.eval()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
