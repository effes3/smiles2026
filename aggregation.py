"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


# Index 0 = token embeddings; indices 1..24 = transformer block outputs for
# Qwen2.5-0.5B. "Dual-core": semantic peak (12) + decision peak (23) last tokens,
# plus norm ratio ||h23_last|| / ||h22_last||.
_SEMANTIC_LAYER: int = 12
_DECISION_LAYER: int = 23
_NORM_RATIO_REFERENCE_LAYER: int = 22


def _resolve_layer_index(selected_layer: int, n_layers: int) -> int:
    layer_idx = selected_layer
    if layer_idx < 0:
        layer_idx += n_layers
    if not 0 <= layer_idx < n_layers:
        raise ValueError(
            f"layer index {selected_layer} invalid for n_layers={n_layers}"
        )
    return layer_idx


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        Concatenation of last-token ``h12``, last-token ``h23``, and scalar
        ``||h23||_2 / ||h22||_2`` → shape ``(2 * hidden_dim + 1,)`` (1793 for
        hidden_dim=896).

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    n_layers = hidden_states.shape[0]
    idx12 = _resolve_layer_index(_SEMANTIC_LAYER, n_layers)
    idx23 = _resolve_layer_index(_DECISION_LAYER, n_layers)
    idx22 = _resolve_layer_index(_NORM_RATIO_REFERENCE_LAYER, n_layers)

    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    h12_last = hidden_states[idx12, last_pos]
    h23_last = hidden_states[idx23, last_pos]
    h22_last = hidden_states[idx22, last_pos]

    norm23 = torch.linalg.norm(h23_last, ord=2)
    norm22 = torch.linalg.norm(h22_last, ord=2)
    norm_ratio = norm23 / (norm22 + torch.finfo(h23_last.dtype).eps)

    return torch.cat([h12_last, h23_last, norm_ratio.reshape(1)], dim=0)
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
