"""Shared WikiText packing / layer-window helpers for scripts/wikitext."""

from __future__ import annotations

import torch
from datasets import load_dataset

SWAP_MODES = ("none", "dark", "middle")


def layer_windows(
    layer_start: int,
    layer_end: int,
    size: int,
    stride: int,
) -> list[list[int]]:
    """Fixed-size sliding windows covering ``[layer_start, layer_end]`` inclusive."""
    if size <= 0:
        raise ValueError(f"--size must be positive, got {size}")
    if stride <= 0:
        raise ValueError(f"--stride must be positive, got {stride}")
    if layer_end < layer_start:
        raise ValueError(
            f"empty interval: layer_start={layer_start} > layer_end={layer_end}"
        )
    span = layer_end - layer_start + 1
    if size > span:
        raise ValueError(
            f"--size={size} exceeds interval length {span} "
            f"([{layer_start}, {layer_end}])"
        )

    last_start = layer_end - size + 1
    starts = list(range(layer_start, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [list(range(s, s + size)) for s in starts]


def mode_label(mode: str) -> str:
    return "none" if mode == "none" else f"{mode}_swap"


def permute_seed_for(base_seed: int, chunk_idx: int, seq_len: int) -> int:
    return int((base_seed + chunk_idx * 1_000_003 + seq_len * 97) % (2**31 - 1))


def build_wikitext_chunks(
    tokenizer,
    *,
    dataset: str,
    dataset_config: str,
    split: str,
    seq_len: int,
    max_chunks: int,
) -> torch.Tensor:
    """Return ``[n_chunks, seq_len]`` token ids (long).

    Tokens are concatenated into one stream, then sliced into **contiguous,
    non-overlapping** windows of length ``seq_len`` (no stride overlap).
    """
    ds = load_dataset(dataset, dataset_config, split=split)
    texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
    if not texts:
        raise ValueError(f"No non-empty texts in {dataset}/{dataset_config}/{split}")

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    stream: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        if bos_id is not None and (not stream or stream[-1] != bos_id):
            stream.append(int(bos_id))
        stream.extend(ids)
        if eos_id is not None:
            stream.append(int(eos_id))
        if max_chunks > 0 and len(stream) >= max_chunks * seq_len:
            break

    if len(stream) < seq_len:
        raise ValueError(
            f"Not enough tokens ({len(stream)}) for seq_len={seq_len}"
        )

    # Non-overlapping pack: consecutive blocks of exactly ``seq_len`` tokens.
    n = len(stream) // seq_len
    if max_chunks > 0:
        n = min(n, max_chunks)
    if n < 1:
        raise ValueError("No WikiText chunks produced")
    flat = torch.tensor(stream[: n * seq_len], dtype=torch.long)
    return flat.view(n, seq_len)
