"""
aggregation.py — Multi-layer feature extraction from hidden states.

Features (~1050 dim):
A. Last-token of final layer (896)
B. Per-layer geometric: norms, means, stds, max-abs (100)
C. Inter-layer cosine similarities (24)
D. Token-level mean-pooling features (6)
E. Positional contrast (2)
F. Cross-layer summary statistics (10)
G. Logit-based features: entropy, max_prob, top_gap (8)
"""

from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

_LM_HEAD_W = None


def _get_lm_head() -> torch.Tensor:
    global _LM_HEAD_W
    if _LM_HEAD_W is None:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        _LM_HEAD_W = m.lm_head.weight.data.float().cpu()
        del m
        gc.collect()
    return _LM_HEAD_W


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
    n_real_int = int(n_real.item())
    mask_float = attention_mask.unsqueeze(-1).float()
    mask_bool = attention_mask.bool()

    features = []

    # A. Last-token of final layer (896)
    last_token_per_layer = hidden_states[:, last_pos, :]
    features.append(hidden_states[-1, last_pos])

    # B. Per-layer geometric features (100)
    layer_norms = torch.norm(last_token_per_layer, p=2, dim=1)
    layer_means = last_token_per_layer.mean(dim=1)
    layer_stds = last_token_per_layer.std(dim=1)
    layer_maxabs = last_token_per_layer.abs().max(dim=1).values
    features.extend([layer_norms, layer_means, layer_stds, layer_maxabs])

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
    features.append(F.cosine_similarity(h_first.unsqueeze(0), h_last.unsqueeze(0)))
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

    # G. Logit-based features (8)
    W = _get_lm_head()
    N = min(10, n_real_int)
    start_tok = max(last_pos - N + 1, first_pos)
    last_n_hidden = hidden_states[-1, start_tok : last_pos + 1, :]

    with torch.no_grad():
        logits = last_n_hidden @ W.T
        log_probs = F.log_softmax(logits, dim=-1)

        max_log_prob = log_probs.max(dim=-1).values
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)

        sorted_logits, _ = logits.sort(dim=-1, descending=True)
        top_gap = sorted_logits[:, 0] - sorted_logits[:, 1]

    features.append(max_log_prob.mean().unsqueeze(0))
    features.append(max_log_prob.min().unsqueeze(0))
    features.append(max_log_prob.std().unsqueeze(0))
    features.append(entropy.mean().unsqueeze(0))
    features.append(entropy.max().unsqueeze(0))
    features.append(entropy.std().unsqueeze(0))
    features.append(top_gap.mean().unsqueeze(0))
    features.append(top_gap.min().unsqueeze(0))

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
