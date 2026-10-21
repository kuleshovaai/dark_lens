"""Build prompt+response token sequences and response-related positions."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def encode_prompt_response(
    tl_model: HookedTransformer,
    prompt: str,
    response: str,
    *,
    prediction_aligned: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return ``(input_ids, response_positions, response_len, seq_len)``.

    Tokenizes prompt and response **separately**, then concatenates IDs so the
    response span is exact (avoids non-compositional string tokenization).

    If ``prediction_aligned`` is True, ``response_positions`` are the residual
    indices whose **next-token logits** predict the response tokens, i.e.
    ``[prompt_len - 1, ..., prompt_len + response_len - 2]``. Use this for
    metrics on predictive distributions (e.g. logit-lens JSD).

    If False (default for dark matter), positions are the indices of the
    response tokens themselves: ``[prompt_len, ..., prompt_len + response_len - 1]``.
    """
    prompt_ids = tl_model.to_tokens(prompt, prepend_bos=False)
    response_ids = tl_model.to_tokens(response, prepend_bos=False)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)

    prompt_len = prompt_ids.shape[1]
    response_len = response_ids.shape[1]
    seq_len = input_ids.shape[1]
    device = tl_model.cfg.device

    if response_len <= 0:
        raise ValueError("Empty response tokenization")
    if prediction_aligned:
        if prompt_len < 1:
            raise ValueError(
                "prediction_aligned positions need at least one prompt token "
                "(last prompt position predicts the first response token)"
            )
        # Positions whose logits predict response tokens r0..r_{R-1}.
        start = prompt_len - 1
        end = prompt_len + response_len - 1  # exclusive
    else:
        start = prompt_len
        end = prompt_len + response_len

    response_positions = torch.arange(start, end, device=device)
    return input_ids, response_positions, response_len, seq_len


def response_token_ids(input_ids: torch.Tensor, response_len: int) -> torch.Tensor:
    """Return the response-half IDs of a concatenated prompt+response sequence."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape [1, seq], got {tuple(input_ids.shape)}")
    if response_len <= 0 or response_len > input_ids.shape[1]:
        raise ValueError(f"Invalid response_len={response_len} for seq_len={input_ids.shape[1]}")
    return input_ids[0, -response_len:]


def _as_id_list(token_ids: torch.Tensor | list[int]) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        return [int(x) for x in token_ids.detach().cpu().view(-1).tolist()]
    return [int(x) for x in token_ids]


def char_offsets_for_token_ids(
    tl_model: HookedTransformer,
    text: str,
    token_ids: torch.Tensor | list[int],
) -> list[tuple[int, int]] | None:
    """Character spans in ``text`` for ``token_ids``, or None if they do not match.

    Prefers the fast tokenizer's ``offset_mapping`` when those IDs are exactly
    ``token_ids``. Otherwise walks decoded token strings if they reconstruct
    ``text`` exactly.
    """
    ids = _as_id_list(token_ids)
    tokenizer = getattr(tl_model, "tokenizer", None)
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            hf_ids = _as_id_list(encoded["input_ids"])
            offsets = encoded.get("offset_mapping") or []
            if hf_ids == ids and len(offsets) == len(ids):
                return [(int(start), int(end)) for start, end in offsets]
        except Exception:
            pass

    ids_tensor = (
        token_ids.detach().cpu().view(-1)
        if isinstance(token_ids, torch.Tensor)
        else torch.tensor(ids, dtype=torch.long)
    )
    str_tokens = tl_model.to_str_tokens(ids_tensor)
    if "".join(str_tokens) == text and len(str_tokens) == len(ids):
        pos = 0
        offsets = []
        for tok in str_tokens:
            offsets.append((pos, pos + len(tok)))
            pos += len(tok)
        return offsets
    return None
