"""Unembedding null-space projection and residual light-matter ratio."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from .tokens import encode_prompt_response, response_token_ids


def ensure_tl_unembedding_shape(w_u: torch.Tensor) -> torch.Tensor:
    """Return W_U in transformer_lens shape [d_model, d_vocab].

    HF ``lm_head.weight`` is [d_vocab, d_model]; transformer_lens ``W_U`` is the transpose.
    """
    w = w_u.detach()
    if w.shape[0] > w.shape[1]:
        w = w.T
    return w


def unembedding_null_basis(w_u: torch.Tensor, null_dim: int = 32) -> torch.Tensor:
    """Orthonormal basis ``[d_model, null_dim]`` for the trailing right singular vectors of ``W_U``."""
    w_u = ensure_tl_unembedding_shape(w_u).detach().cpu().float()
    _, _, vh = torch.linalg.svd(w_u.T, full_matrices=False)
    return vh[-null_dim:, :].T.contiguous()


def unembedding_top_basis(w_u: torch.Tensor, null_dim: int = 32) -> torch.Tensor:
    """Orthonormal basis ``[d_model, null_dim]`` for the leading right singular vectors of ``W_U``."""
    w_u = ensure_tl_unembedding_shape(w_u).detach().cpu().float()
    _, _, vh = torch.linalg.svd(w_u.T, full_matrices=False)
    return vh[:null_dim, :].T.contiguous()


def unembedding_mid_basis(w_u: torch.Tensor, null_dim: int = 32) -> torch.Tensor:
    """Orthonormal basis ``[d_model, null_dim]`` from the middle of the ``W_U`` SVD.

    Right singular vectors are ordered by descending singular value (same as
    ``unembedding_top_basis`` / ``unembedding_null_basis``). This returns the
    same-rank slice centered in that spectrum: start at ``(n - k) // 2``.
    """
    vh = unembedding_right_basis(w_u)
    n_sv = vh.shape[0]
    if null_dim <= 0 or null_dim > n_sv:
        raise ValueError(f"null_dim={null_dim} must be in 1..{n_sv}")
    start = (n_sv - null_dim) // 2
    return vh[start : start + null_dim, :].T.contiguous()


def unembedding_right_basis(w_u: torch.Tensor) -> torch.Tensor:
    """Full right-singular basis ``[n_sv, d_model]`` of ``W_U`` (descending σ)."""
    w_u = ensure_tl_unembedding_shape(w_u).detach().cpu().float()
    _, _, vh = torch.linalg.svd(w_u.T, full_matrices=False)
    return vh.contiguous()


def mean_subspace_norm_fraction(
    hidden: torch.Tensor,
    basis: torch.Tensor,
) -> float:
    """Mean ``||P_V h|| / ||h||`` over leading axes of ``hidden``.

    ``basis`` is ``[d_model, k]`` with orthonormal columns. ``hidden`` is
    ``[..., d_model]`` (flattened over all but the last dim).
    """
    h = hidden.reshape(-1, hidden.shape[-1]).float()
    if h.numel() == 0:
        raise ValueError("hidden is empty")
    v = basis.to(device=h.device, dtype=torch.float32)
    norms = torch.linalg.norm(h, dim=-1).clamp_min(1e-8)
    removed = torch.linalg.norm(h @ v, dim=-1)
    return float((removed / norms).mean().item())


def middle_rank_matching_norm_fraction(
    hidden: torch.Tensor,
    vh: torch.Tensor,
    target_frac: float,
) -> tuple[int, float]:
    """Mid-band rank whose mean ``||P h||/||h||`` is closest to ``target_frac``.

    ``vh`` is ``[n_sv, d_model]`` from ``unembedding_right_basis``. The mid band
    of rank ``k`` is the centered slice ``vh[(n-k)//2 : (n-k)//2 + k]``.
    Returns ``(k, achieved_frac)``.
    """
    if target_frac < 0:
        raise ValueError(f"target_frac must be >= 0, got {target_frac}")
    h = hidden.reshape(-1, hidden.shape[-1]).float()
    if h.numel() == 0:
        raise ValueError("hidden is empty")
    vh = vh.to(device=h.device, dtype=torch.float32)
    n_sv = vh.shape[0]
    norms = torch.linalg.norm(h, dim=-1).clamp_min(1e-8)
    # Energy along each singular direction: [N, n_sv]
    energy = (h @ vh.mT).pow(2)

    def frac_for(k: int) -> float:
        start = (n_sv - k) // 2
        rem = energy[:, start : start + k].sum(dim=-1).clamp_min(0.0).sqrt()
        return float((rem / norms).mean().item())

    # Smallest k with frac >= target (band energy grows with k), then pick
    # between that k and k-1 by absolute error.
    lo, hi = 1, n_sv
    cross = n_sv
    while lo <= hi:
        mid = (lo + hi) // 2
        if frac_for(mid) + 1e-12 >= target_frac:
            cross = mid
            hi = mid - 1
        else:
            lo = mid + 1
    candidates = {cross}
    if cross > 1:
        candidates.add(cross - 1)
    best_k = min(candidates, key=lambda k: abs(frac_for(k) - target_frac))
    return best_k, frac_for(best_k)


def random_null_basis(
    d_model: int,
    null_dim: int = 32,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Haar-random orthonormal basis ``[d_model, null_dim]``."""
    if null_dim <= 0 or null_dim > d_model:
        raise ValueError(f"null_dim={null_dim} must be in 1..{d_model}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn(d_model, null_dim, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(gaussian, mode="reduced")
    return q.contiguous()


def build_null_space_projection(w_u: torch.Tensor, null_dim: int = 32) -> torch.Tensor:
    """Projection matrix onto the null space of W_U (smallest singular values).

    Uses SVD on W_U.T so right singular vectors live in hidden space (d_model).
    """
    v_null = unembedding_null_basis(w_u, null_dim)
    return (v_null @ v_null.T).detach()


def build_top_space_projection(w_u: torch.Tensor, null_dim: int = 32) -> torch.Tensor:
    """Projection onto the leading ``null_dim`` right singular vectors of W_U.

    Same rank as ``build_null_space_projection``, but the subspace is the most
    logit-aligned directions (largest singular values) rather than the trailing
    null space.
    """
    v_top = unembedding_top_basis(w_u, null_dim)
    return (v_top @ v_top.T).detach()


def build_mid_space_projection(w_u: torch.Tensor, null_dim: int = 32) -> torch.Tensor:
    """Projection onto a centered band of ``null_dim`` right singular vectors of W_U.

    Same rank as ``build_null_space_projection``, but the subspace is the middle
    of the unembedding spectrum rather than the trailing null space.
    """
    v_mid = unembedding_mid_basis(w_u, null_dim)
    return (v_mid @ v_mid.T).detach()


def build_random_null_space_projection(
    d_model: int,
    null_dim: int = 32,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Rank-``null_dim`` orthogonal projector onto a random subspace of R^{d_model}.

    Same rank as ``build_null_space_projection``, but the subspace is Haar-random
    rather than the trailing right singular vectors of W_U.
    """
    q = random_null_basis(d_model, null_dim, seed=seed)
    return (q @ q.T).detach()


def project_out_and_renormalize(
    hidden: torch.Tensor,
    p_remove: torch.Tensor,
) -> torch.Tensor:
    """Remove a subspace from ``hidden`` and restore the original L2 norm.

    ``p_remove`` is either a ``[d_model, d_model]`` projector or a
    ``[d_model, k]`` orthonormal basis. Each vector ``h`` is replaced by
    ``(I - P) h``, then rescaled so ``||h'|| = ||h||``.
    Arithmetic is in float32, then cast back to ``hidden.dtype``.
    """
    p = p_remove.to(device=hidden.device, dtype=torch.float32)
    h = hidden.float()
    orig_norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(1e-8)
    if p.shape[-1] == p.shape[-2]:
        h_kept = h - h @ p
    else:
        h_kept = h - (h @ p) @ p.mT
    kept_norm = torch.linalg.norm(h_kept, dim=-1, keepdim=True).clamp_min(1e-8)
    return (h_kept * (orig_norm / kept_norm)).to(dtype=hidden.dtype)


def reweight_subspace_and_renormalize(
    hidden: torch.Tensor,
    p_subspace: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Boost a subspace vs its complement, then restore the original L2 norm.

    Split ``h = h_sub + h_comp`` using ``p_subspace`` (a ``[d_model, d_model]``
    projector or ``[d_model, k]`` orthonormal basis) and replace it by

        ``ĥ = (1 + α) h_sub + (1 - α) h_comp``

    then rescale so ``||ĥ|| = ||h||``. ``α = 0`` is a no-op before
    renormalize; ``α > 0`` enhances the subspace (dark or middle).
    """
    p = p_subspace.to(device=hidden.device, dtype=torch.float32)
    h = hidden.float()
    orig_norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(1e-8)
    if p.shape[-1] == p.shape[-2]:
        h_sub = h @ p
    else:
        h_sub = (h @ p) @ p.mT
    h_hat = (1.0 + alpha) * h_sub + (1.0 - alpha) * (h - h_sub)
    hat_norm = torch.linalg.norm(h_hat, dim=-1, keepdim=True).clamp_min(1e-8)
    return (h_hat * (orig_norm / hat_norm)).to(dtype=hidden.dtype)


def swap_subspace_and_renormalize(
    hidden: torch.Tensor,
    p_subspace: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """Replace each vector's subspace component with that of ``hidden[perm]``.

    ``perm`` indexes the batch axis. ``p_subspace`` is a ``[d_model, d_model]``
    projector or a ``[d_model, k]`` orthonormal basis. The receiver's original
    L2 norm is restored, matching ``project_out_and_renormalize``.
    """
    p = p_subspace.to(device=hidden.device, dtype=torch.float32)
    h = hidden.float()
    orig_norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(1e-8)
    if p.shape[-1] == p.shape[-2]:
        dark = h @ p
    else:
        dark = (h @ p) @ p.mT
    donor_dark = dark.index_select(0, perm.to(device=h.device, dtype=torch.long))
    h_new = h - dark + donor_dark
    new_norm = torch.linalg.norm(h_new, dim=-1, keepdim=True).clamp_min(1e-8)
    return (h_new * (orig_norm / new_norm)).to(dtype=hidden.dtype)


def permute_subspace_and_renormalize(
    hidden: torch.Tensor,
    p_subspace: torch.Tensor,
    perm: torch.Tensor,
    *,
    dim: int = 1,
) -> torch.Tensor:
    """Permute subspace components along ``dim``; leave the complement in place.

    ``perm`` indexes ``dim`` (default: sequence). For an in-sample token shuffle
    of shape ``[batch, seq, d_model]``, pass a length-``seq`` permutation.
    The receiver's original L2 is restored, matching ``swap_subspace_and_renormalize``.
    """
    p = p_subspace.to(device=hidden.device, dtype=torch.float32)
    h = hidden.float()
    orig_norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(1e-8)
    if p.shape[-1] == p.shape[-2]:
        dark = h @ p
    else:
        dark = (h @ p) @ p.mT
    donor_dark = dark.index_select(
        dim, perm.to(device=h.device, dtype=torch.long)
    )
    h_new = h - dark + donor_dark
    new_norm = torch.linalg.norm(h_new, dim=-1, keepdim=True).clamp_min(1e-8)
    return (h_new * (orig_norm / new_norm)).to(dtype=hidden.dtype)


def light_matter_ratio(hidden: torch.Tensor, p_null: torch.Tensor) -> torch.Tensor:
    """Per-vector light matter ratio 1 - ||P_null h|| / ||h||."""
    h = hidden.float()
    p = p_null.float().to(device=h.device)
    projected = h @ p
    null_norm = torch.linalg.norm(projected, dim=-1)
    h_norm = torch.linalg.norm(h, dim=-1)
    return 1 - null_norm / h_norm.clamp_min(1e-8)


def aggregate_masked(
    values: torch.Tensor,
    mask: torch.Tensor,
    aggregation: Literal["mean", "min", "max", "last"] = "mean",
) -> torch.Tensor:
    """Aggregate per-position values over masked positions."""
    mask = mask.bool().to(values.device)
    if aggregation == "mean":
        masked = values * mask
        return masked.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    if aggregation == "min":
        masked = values.masked_fill(~mask, float("inf"))
        return masked.min(dim=-1).values
    if aggregation == "max":
        masked = values.masked_fill(~mask, float("-inf"))
        return masked.max(dim=-1).values
    if aggregation == "last":
        last_idx = mask.long().sum(dim=-1) + torch.argmax(mask.long(), dim=-1) - 1
        batch_idx = torch.arange(values.size(0), device=values.device)
        return values[batch_idx, last_idx]
    raise ValueError(f"Unsupported aggregation method: {aggregation}")


def light_matter_ratio_masked(
    hiddens: torch.Tensor,
    mask: torch.Tensor,
    p_null: torch.Tensor,
    aggregation: Literal["mean", "min", "max", "last"] = "mean",
) -> torch.Tensor:
    """Compute light matter ratio per token, then aggregate over masked positions.

    Matches the CoQA / RAGTruth layer-wise compute scripts when ``aggregation='mean'``
    and ``mask`` selects response tokens.
    """
    ratios = light_matter_ratio(hiddens, p_null)
    return aggregate_masked(ratios, mask, aggregation)


def light_matter_ratio_per_layer(
    hidden_by_layer: torch.Tensor,
    response_positions: torch.Tensor,
    p_null: torch.Tensor,
) -> torch.Tensor:
    """Mean light matter ratio over response tokens, per layer.

    Args:
        hidden_by_layer: [n_layers, seq_len, d_model]
        response_positions: indices of response tokens
        p_null: [d_model, d_model]

    Returns:
        Tensor of shape [n_layers].
    """
    h = hidden_by_layer[:, response_positions, :]
    ratios = light_matter_ratio(h, p_null)
    return ratios.mean(dim=-1)


def light_matter_token_curve(
    tl_model: HookedTransformer,
    prompt: str,
    response: str,
    p_null: torch.Tensor,
    *,
    prediction_aligned: bool = False,
) -> tuple[np.ndarray, int, int, torch.Tensor]:
    """Per-token light-matter ratio at response-related residual states, per layer.

    Defaults to token-presence positions (not prediction-aligned).

    Returns:
        ratios: ``[n_layers, n_response_positions]``
        response_len, seq_len
        response_ids: 1-D tensor of the response token IDs used in the forward pass
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
        return name.endswith(".hook_resid_post")

    with torch.no_grad():
        _, cache = tl_model.run_with_cache(inputs_total, names_filter=names_filter)

    n_layers = tl_model.cfg.n_layers
    hidden_by_layer = torch.stack(
        [cache[f"blocks.{layer}.hook_resid_post"][0] for layer in range(n_layers)],
        dim=0,
    )
    ratios = light_matter_ratio(hidden_by_layer[:, response_positions, :], p_null)
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ratios.cpu().numpy(), response_len, seq_len, response_ids


def light_matter_curve(
    tl_model: HookedTransformer,
    prompt: str,
    response: str,
    p_null: torch.Tensor,
    *,
    prediction_aligned: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Mean light-matter ratio over response-token residual states, per layer.

    Defaults to token-presence positions (not prediction-aligned): light matter is
    a geometric property of residual vectors at response tokens, not of
    next-token predictive distributions.
    """
    ratios, response_len, seq_len, _ = light_matter_token_curve(
        tl_model,
        prompt,
        response,
        p_null,
        prediction_aligned=prediction_aligned,
    )
    return ratios.mean(axis=-1), response_len, seq_len


def light_matter_token_curve_detailed(
    tl_model: HookedTransformer,
    prompt: str,
    response: str,
    p_null: torch.Tensor,
    *,
    prediction_aligned: bool = False,
) -> tuple[np.ndarray, int, int, torch.Tensor]:
    """Per-token light-matter ratio at response-related residual states, per layer.

    Defaults to token-presence positions (not prediction-aligned).

    Returns:
        ratios: ``[2 * n_layers, n_response_positions]``
        response_len, seq_len
        response_ids: 1-D tensor of the response token IDs used in the forward pass
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
        return name.endswith(".hook_resid_post") or name.endswith(".hook_resid_mid")

    with torch.no_grad():
        _, cache = tl_model.run_with_cache(inputs_total, names_filter=names_filter)

    n_layers = tl_model.cfg.n_layers

    hidden_by_layer = torch.concatenate(
        [torch.stack((cache[f"blocks.{layer}.hook_resid_mid"][0], cache[f"blocks.{layer}.hook_resid_post"][0]), dim=0) for layer in range(n_layers)],
        dim=0,
    )
    ratios = light_matter_ratio(hidden_by_layer[:, response_positions, :], p_null)
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ratios.cpu().numpy(), response_len, seq_len, response_ids
