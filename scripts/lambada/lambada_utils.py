"""Shared LAMBADA packing / layer-window helpers for scripts/lambada.

Defaults to ``EleutherAI/lambada_openai`` / ``en`` / ``test``. Same BOS/EOS
non-overlapping packing and pairwise-swap helpers as WikiText.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import load_dataset

# Reuse window / mode helpers from the WikiText scripts.
_WIKITEXT_DIR = Path(__file__).resolve().parent.parent / "wikitext"
if str(_WIKITEXT_DIR) not in sys.path:
    sys.path.insert(0, str(_WIKITEXT_DIR))

from wikitext_utils import (  # noqa: E402, F401
    SWAP_MODES,
    layer_windows,
    mode_label,
    permute_seed_for,
)

DEFAULT_LAMBADA_DATASET = "EleutherAI/lambada_openai"
DEFAULT_LAMBADA_CONFIG = "en"
DEFAULT_LAMBADA_SPLIT = "test"


def build_lambada_chunks(
    tokenizer,
    *,
    dataset: str = DEFAULT_LAMBADA_DATASET,
    dataset_config: str | None = DEFAULT_LAMBADA_CONFIG,
    split: str = DEFAULT_LAMBADA_SPLIT,
    seq_len: int,
    max_chunks: int,
    text_field: str = "text",
) -> torch.Tensor:
    """Return ``[n_chunks, seq_len]`` token ids packed from LAMBADA.

    Tokens are concatenated into one stream, then sliced into contiguous,
    non-overlapping windows of length ``seq_len`` (same as WikiText).
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")

    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split)
    else:
        ds = load_dataset(dataset, split=split)

    texts = [
        t
        for t in ds[text_field]
        if isinstance(t, str) and t.strip()
    ]
    if not texts:
        raise ValueError(
            f"No non-empty texts in {dataset}"
            + (f"/{dataset_config}" if dataset_config else "")
            + f"/{split}"
        )

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
            f"Not enough tokens ({len(stream)}) for seq_len={seq_len} from "
            f"{dataset}"
            + (f"/{dataset_config}" if dataset_config else "")
            + f"/{split}"
        )

    n = len(stream) // seq_len
    if max_chunks > 0:
        n = min(n, max_chunks)
    if n < 1:
        raise ValueError("No LAMBADA chunks produced")
    flat = torch.tensor(stream[: n * seq_len], dtype=torch.long)
    return flat.view(n, seq_len)


def dataset_tag(dataset: str, dataset_config: str | None, split: str) -> str:
    """Short filesystem-safe tag for output filenames."""
    parts = [dataset.replace("/", "_")]
    if dataset_config:
        parts.append(dataset_config.replace("/", "_"))
    parts.append(split)
    return "_".join(parts)
