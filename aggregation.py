"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Configuration aligned with ``tests/TESTS_Ratio.md`` (~73.77% mean test AUROC,
5-fold, seeds 42–44): last token at layer **23**, **‖h23_last‖/‖h22_last‖**,
cosine **(h23_last, h22_last)**, and **‖h23_last − h23_first‖** on layer 23.
"""

from __future__ import annotations

import torch

_FEATURE_LAYER: int = 23
_NORM_RATIO_REFERENCE_LAYER: int = 22


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Last-token ``h23`` plus norm ratio, L23–L22 last cosine, L23 first–last distance."""
    n_layers = hidden_states.shape[0]

    def _idx(ell: int) -> int:
        i = ell if ell >= 0 else ell + n_layers
        if not 0 <= i < n_layers:
            raise ValueError(f"layer {ell} invalid for n_layers={n_layers}")
        return i

    idx23 = _idx(_FEATURE_LAYER)
    idx22 = _idx(_NORM_RATIO_REFERENCE_LAYER)

    real_positions = attention_mask.nonzero(as_tuple=False)
    first_pos = int(real_positions[0].item())
    last_pos = int(real_positions[-1].item())

    layer = hidden_states[idx23]
    reference_layer = hidden_states[idx22]
    first_state = layer[first_pos]
    last_state = layer[last_pos]
    reference_last_state = reference_layer[last_pos]

    eps = torch.finfo(last_state.dtype).eps
    reference_norm = torch.linalg.norm(reference_last_state, ord=2)
    last_norm = torch.linalg.norm(last_state, ord=2)
    norm_ratio = last_norm / (reference_norm + eps)
    cosine_similarity = torch.dot(last_state, reference_last_state) / (
        last_norm * reference_norm + eps
    )
    relative_distance = torch.linalg.norm(last_state - first_state, ord=2)

    geometry = torch.stack([norm_ratio, cosine_similarity, relative_distance])
    return torch.cat([last_state, geometry], dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
