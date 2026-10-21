"""Shared CCNet packing helpers for scripts/ccnet.

Defaults to ``JorgeeGF/CCNet`` / ``train``, loaded with streaming. Same BOS/EOS
packing as WikiText / C4 / Pile. Text field is ``raw_content``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import load_dataset

_WIKITEXT_DIR = Path(__file__).resolve().parent.parent / "wikitext"
if str(_WIKITEXT_DIR) not in sys.path:
    sys.path.insert(0, str(_WIKITEXT_DIR))

from wikitext_utils import (  # noqa: E402, F401
    SWAP_MODES,
    layer_windows,
    mode_label,
    permute_seed_for,
)

DEFAULT_CCNET_DATASET = "JorgeeGF/CCNet"
DEFAULT_CCNET_CONFIG = None
DEFAULT_CCNET_SPLIT = "train"
DEFAULT_CCNET_TEXT_FIELD = "raw_content"


def build_ccnet_chunks(
    tokenizer,
    *,
    dataset: str = DEFAULT_CCNET_DATASET,
    dataset_config: str | None = DEFAULT_CCNET_CONFIG,
    split: str = DEFAULT_CCNET_SPLIT,
    seq_len: int,
    max_chunks: int,
    streaming: bool = True,
    text_field: str = DEFAULT_CCNET_TEXT_FIELD,
) -> torch.Tensor:
    """Return ``[n_chunks, seq_len]`` token ids packed from CCNet.

    Streams documents until ``max_chunks * seq_len`` tokens are collected (or
    the stream ends). Same BOS/EOS packing convention as WikiText / C4 / Pile.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")
    if max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")

    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset, split=split, streaming=streaming)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    need = max_chunks * seq_len
    stream: list[int] = []

    for row in ds:
        text = row.get(text_field) if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        if bos_id is not None and (not stream or stream[-1] != bos_id):
            stream.append(int(bos_id))
        stream.extend(ids)
        if eos_id is not None:
            stream.append(int(eos_id))
        if len(stream) >= need:
            break

    if len(stream) < seq_len:
        raise ValueError(
            f"Not enough tokens ({len(stream)}) for seq_len={seq_len} from "
            f"{dataset}"
            + (f"/{dataset_config}" if dataset_config else "")
            + f"/{split}"
        )

    n = min(len(stream) // seq_len, max_chunks)
    if n < 1:
        raise ValueError("No CCNet chunks produced")
    flat = torch.tensor(stream[: n * seq_len], dtype=torch.long)
    return flat.view(n, seq_len)


def dataset_tag(dataset: str, dataset_config: str | None, split: str) -> str:
    """Short filesystem-safe tag for output filenames."""
    parts = [dataset.replace("/", "_")]
    if dataset_config:
        parts.append(dataset_config.replace("/", "_"))
    parts.append(split)
    return "_".join(parts)
