"""FFN logit-lens Jensen–Shannon divergence on residual-stream states."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from .tokens import encode_prompt_response, response_token_ids


def jensen_shannon_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def logit_lens_jsd_by_token(
    ffn_input_by_layer: torch.Tensor,
    ffn_output_by_layer: torch.Tensor,
    response_positions: torch.Tensor,
    w_u: torch.Tensor,
) -> torch.Tensor:
    h_in = ffn_input_by_layer[:, response_positions, :].float()
    h_out = ffn_output_by_layer[:, response_positions, :].float()
    probs_in = torch.softmax(h_in @ w_u.float(), dim=-1)
    probs_out = torch.softmax(h_out @ w_u.float(), dim=-1)
    return jensen_shannon_divergence(probs_in, probs_out)


def logit_lens_jsd_per_layer(
    ffn_input_by_layer: torch.Tensor,
    ffn_output_by_layer: torch.Tensor,
    response_positions: torch.Tensor,
    w_u: torch.Tensor,
) -> torch.Tensor:
    return logit_lens_jsd_by_token(
        ffn_input_by_layer, ffn_output_by_layer, response_positions, w_u
    ).mean(dim=-1)


def jsd_token_curve(
    tl_model: HookedTransformer,
    prompt: str,
    response: str,
    w_u: torch.Tensor,
    *,
    prediction_aligned: bool = True,
) -> tuple[np.ndarray, int, int, torch.Tensor]:
    """Per-token FFN in/out logit-lens JSD at response-related positions, per layer.

    Defaults to prediction-aligned positions. Returns
    ``(jsd [n_layers, n_positions], response_len, seq_len, response_ids)``.
    ``response_ids`` are the generated-token IDs (not the residual positions).
    """
    response_text = "" if pd.isna(response) else str(response)
    prompt_text = "" if pd.isna(prompt) else str(prompt)
    if not response_text.strip():
        raise ValueError("Empty generated_response")

    inputs_total, response_positions, response_len, seq_len = encode_prompt_response(
        tl_model,
        prompt_text,
        response_text,
        prediction_aligned=prediction_aligned,
    )
    response_ids = response_token_ids(inputs_total, response_len)

    def names_filter(name: str) -> bool:
        return name.endswith(".hook_resid_mid") or name.endswith(".hook_resid_post")

    with torch.no_grad():
        _, cache = tl_model.run_with_cache(inputs_total, names_filter=names_filter)

    n_layers = tl_model.cfg.n_layers
    ffn_input = torch.stack(
        [cache[f"blocks.{layer}.hook_resid_mid"][0] for layer in range(n_layers)],
        dim=0,
    )
    ffn_output = torch.stack(
        [cache[f"blocks.{layer}.hook_resid_post"][0] for layer in range(n_layers)],
        dim=0,
    )
    jsd = logit_lens_jsd_by_token(ffn_input, ffn_output, response_positions, w_u)
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return jsd.cpu().numpy(), response_len, seq_len, response_ids


def jsd_curve(
    tl_model: HookedTransformer,
    prompt: str,
    response: str,
    w_u: torch.Tensor,
    *,
    prediction_aligned: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Mean FFN in/out logit-lens JSD over response-related positions, per layer."""
    jsd, response_len, seq_len, _ = jsd_token_curve(
        tl_model,
        prompt,
        response,
        w_u,
        prediction_aligned=prediction_aligned,
    )
    return jsd.mean(axis=-1), response_len, seq_len
