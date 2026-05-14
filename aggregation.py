"""
aggregation.py — Multi-layer feature extraction from hidden states.

Features (~1038 dim):
- Last-token of final layer (896)
- Per-layer geometric: norms, means, stds, max-abs (100)
- Inter-layer cosine similarities (24)
- Token-level mean-pooling features (6)
- Positional contrast (2)
- Cross-layer summary statistics (10)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    n_layers = hidden_states.shape[0]
    hidden_states = hidden_states.cpu()
    attention_mask = attention_mask.cpu()

    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())
    first_pos = int(real_positions[0].item())
    n_real = attention_mask.sum().float().clamp(min=1.0)
    mask_float = attention_mask.unsqueeze(-1).float()

    features = []

    # A. Last-token of final layer (896)
    features.append(hidden_states[-1, last_pos])

    # B. Per-layer geometric features (100)
    last_token_per_layer = hidden_states[:, last_pos, :]
    layer_norms = torch.norm(last_token_per_layer, p=2, dim=1)
    layer_means = last_token_per_layer.mean(dim=1)
    layer_stds = last_token_per_layer.std(dim=1)
    layer_maxabs = last_token_per_layer.abs().max(dim=1).values

    features.append(layer_norms)
    features.append(layer_means)
    features.append(layer_stds)
    features.append(layer_maxabs)

    # C. Inter-layer cosine similarities (24)
    normed = F.normalize(last_token_per_layer, p=2, dim=1)
    cosine_sims = (normed[:-1] * normed[1:]).sum(dim=1)
    features.append(cosine_sims)

    # D. Token-level mean-pooled features for layers 0, 12, 24 (6)
    for layer_idx in [0, 12, min(n_layers - 1, 24)]:
        layer_hidden = hidden_states[layer_idx]
        mp = (layer_hidden * mask_float).sum(dim=0) / n_real
        features.append(torch.norm(mp, p=2).unsqueeze(0))
        cos = F.cosine_similarity(
            last_token_per_layer[layer_idx].unsqueeze(0),
            mp.unsqueeze(0),
        )
        features.append(cos)

    # E. Positional contrast (2)
    h_first = hidden_states[-1, first_pos]
    h_last = hidden_states[-1, last_pos]
    features.append(
        F.cosine_similarity(h_first.unsqueeze(0), h_last.unsqueeze(0))
    )
    features.append(
        (torch.norm(h_last, p=2) / torch.norm(h_first, p=2).clamp(min=1e-8)).unsqueeze(0)
    )

    # F. Cross-layer summary statistics (10)
    features.append(layer_norms.mean().unsqueeze(0))
    features.append(layer_norms.std().unsqueeze(0))
    features.append(layer_norms.max().unsqueeze(0))
    features.append(layer_norms.min().unsqueeze(0))
    features.append(layer_norms.argmax().float().unsqueeze(0))

    norm_diffs = layer_norms[1:] - layer_norms[:-1]
    features.append(norm_diffs.mean().unsqueeze(0))
    norm_curvature = norm_diffs[1:] - norm_diffs[:-1]
    features.append(norm_curvature.mean().unsqueeze(0))

    features.append(cosine_sims.mean().unsqueeze(0))
    features.append(cosine_sims.std().unsqueeze(0))
    features.append(cosine_sims.min().unsqueeze(0))

    return torch.cat([f.flatten().float() for f in features])


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
    return aggregate(hidden_states, attention_mask)
