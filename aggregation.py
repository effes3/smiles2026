"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Spectral-style diagnostics inspired by Noël (2026), arXiv:2602.08082 §3.3
(spectral entropy, HFER, smoothness on a graph Laplacian), combined with
last-token readouts at layers 19, 22, 23 and a norm ratio.

Because ``solution.py`` is fixed and does not expose attention weights, the
graph Laplacian is built from **hidden states only**: an undirected path graph
on real tokens with edge weights ``relu(cos(h_i, h_{i+1})) + eps``. The same
formulas as in the paper (GFT via ``eigh(L)``, energy split, Dirichlet
quotient) are then applied on that proxy topology.
"""

from __future__ import annotations

import torch

_LAYER_SMOOTH: int = 19
_LAYER_HFER_A: int = 17
_LAYER_HFER_B: int = 22
_LAYER_NORM_DENOM: int = 22
_LAYER_NORM_NUM: int = 23
_LAYER_ENTROPY: int = 23


def _resolve_layer_index(selected_layer: int, n_layers: int) -> int:
    layer_idx = selected_layer
    if layer_idx < 0:
        layer_idx += n_layers
    if not 0 <= layer_idx < n_layers:
        raise ValueError(
            f"layer index {selected_layer} invalid for n_layers={n_layers}"
        )
    return layer_idx


def _real_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask != 0).nonzero(as_tuple=False).flatten()


def _crop_hidden(h_seq: torch.Tensor, real_idx: torch.Tensor) -> torch.Tensor:
    return h_seq[real_idx]


def _chain_cosine_graph(X: torch.Tensor) -> torch.Tensor:
    """Path graph on consecutive tokens; weights from cosine similarity (proxy)."""
    X = X.float()
    n, _d = X.shape
    W = torch.zeros(n, n, dtype=X.dtype, device=X.device)
    if n < 2:
        return W
    a = X[:-1]
    b = X[1:]
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
    w = torch.relu(num / den) + 1e-4
    idx = torch.arange(n - 1, device=X.device)
    W[idx, idx + 1] = w
    W[idx + 1, idx] = w
    return W


def _spectral_entropy_hfer_smoothness(
    X: torch.Tensor, W: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spectral entropy, HFER, smoothness (Defs. 3.1, 3.3, 3.4) for graph ``W``."""
    X = X.float()
    W = W.float()
    n, _d = X.shape
    if n < 2:
        z = torch.zeros((), dtype=torch.float32, device=X.device)
        return z, z, z

    deg = W.sum(dim=-1)
    L = torch.diag(deg) - W
    evals, U = torch.linalg.eigh(L)
    x_hat = U.T @ X
    row_e = (x_hat * x_hat).sum(dim=-1).clamp_min(1e-12)
    total = row_e.sum().clamp_min(1e-12)
    p = row_e / total
    se = -(p * torch.log(p.clamp_min(1e-12))).sum()

    hfer = row_e[n // 2 :].sum() / total

    lam_n = evals[-1].abs().clamp_min(1e-12)
    lx = L @ X
    trace_xtlx = (X * lx).sum()
    fro2 = (X * X).sum().clamp_min(1e-12)
    smooth = 1.0 - trace_xtlx / (lam_n * fro2)

    return se, hfer, smooth


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """L19/L22/L23 last tokens + SE/HFER/S on proxy chain graphs + norm ratio."""
    n_layers = hidden_states.shape[0]
    idx19 = _resolve_layer_index(_LAYER_SMOOTH, n_layers)
    idx17 = _resolve_layer_index(_LAYER_HFER_A, n_layers)
    idx22 = _resolve_layer_index(_LAYER_HFER_B, n_layers)
    idx23 = _resolve_layer_index(_LAYER_ENTROPY, n_layers)

    real_idx = _real_indices(attention_mask)
    last_pos = int(real_idx[-1].item())

    h19_last = hidden_states[idx19, last_pos]
    h22_last = hidden_states[idx22, last_pos]
    h23_last = hidden_states[idx23, last_pos]

    eps = torch.finfo(h23_last.dtype).eps
    n22 = torch.linalg.norm(h22_last, ord=2)
    n23 = torch.linalg.norm(h23_last, ord=2)
    norm_ratio = n23 / (n22 + eps)

    x17 = _crop_hidden(hidden_states[idx17], real_idx)
    w17 = _chain_cosine_graph(x17)
    _, hfer17, _ = _spectral_entropy_hfer_smoothness(x17, w17)

    x22 = _crop_hidden(hidden_states[idx22], real_idx)
    w22 = _chain_cosine_graph(x22)
    _, hfer22, _ = _spectral_entropy_hfer_smoothness(x22, w22)

    x23 = _crop_hidden(hidden_states[idx23], real_idx)
    w23 = _chain_cosine_graph(x23)
    se23, _, _ = _spectral_entropy_hfer_smoothness(x23, w23)

    x19 = _crop_hidden(hidden_states[idx19], real_idx)
    w19 = _chain_cosine_graph(x19)
    _, _, smooth19 = _spectral_entropy_hfer_smoothness(x19, w19)

    scalars = torch.stack(
        [
            se23.to(h19_last.dtype),
            hfer17.to(h19_last.dtype),
            hfer22.to(h19_last.dtype),
            smooth19.to(h19_last.dtype),
        ]
    )

    return torch.cat(
        [h19_last, h22_last, h23_last, scalars, norm_ratio.reshape(1)], dim=0
    )


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
