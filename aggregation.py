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
# Qwen2.5-0.5B. Use the late-layer stack that won the K-fold comparison.
_STACK_LAYER_INDICES: tuple[int, ...] = (21, 22, 23)


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
        A 1-D feature tensor containing last real token vectors from layers
        21, 22, and 23, followed by three geometric scalars computed on layer
        23: ``||last||_2``, cosine similarity between first and last real token,
        and Euclidean distance between first and last real token.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    # Last real token from a late-layer stack.
    n_layers = hidden_states.shape[0]
    layer_indices: list[int] = []
    for selected_layer in _STACK_LAYER_INDICES:
        layer_idx = selected_layer
        if layer_idx < 0:
            layer_idx += n_layers
        if not 0 <= layer_idx < n_layers:
            raise ValueError(
                f"layer index {selected_layer} invalid for n_layers={n_layers}"
            )
        layer_indices.append(layer_idx)

    # Find the first/last real (non-padding) positions. The true prompt/response
    # boundary is not available here, so first-vs-last is a no-solution.py proxy
    # for context-to-answer drift.
    real_positions = attention_mask.nonzero(as_tuple=False)  # (n_real, 1)
    first_pos = int(real_positions[0].item())                 # scalar index
    last_pos = int(real_positions[-1].item())                 # scalar index

    stacked_last_states = [
        hidden_states[layer_idx, last_pos] for layer_idx in layer_indices
    ]

    layer = hidden_states[layer_indices[-1]]  # geometry readout layer
    first_state = layer[first_pos]     # (hidden_dim,)
    last_state = layer[last_pos]       # (hidden_dim,)

    magnitude = torch.linalg.norm(last_state, ord=2)
    first_norm = torch.linalg.norm(first_state, ord=2)
    last_norm = torch.linalg.norm(last_state, ord=2)
    cosine_similarity = torch.dot(first_state, last_state) / (
        first_norm * last_norm + torch.finfo(last_state.dtype).eps
    )
    relative_distance = torch.linalg.norm(last_state - first_state, ord=2)

    geometry = torch.stack([magnitude, cosine_similarity, relative_distance])
    return torch.cat([*stacked_last_states, geometry], dim=0)
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
