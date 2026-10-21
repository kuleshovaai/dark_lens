"""Shared The Pile packing helpers for scripts/pile.

Defaults to the uncopyrighted HF mirror ``monology/pile-uncopyrighted`` (train),
loaded with streaming so we never materialize the full corpus.

That mirror ships ``.jsonl.zst`` shards; install ``zstandard`` so fsspec can
decompress them (``pip install zstandard``).
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

DEFAULT_PILE_DATASET = "monology/pile-uncopyrighted"
DEFAULT_PILE_SPLIT = "train"


def _ensure_zstd_support() -> None:
    """``monology/pile-uncopyrighted`` is ``.jsonl.zst``; fsspec needs zstandard."""
    try:
        import zstandard  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading The Pile mirror requires the 'zstandard' package "
            "(files are .jsonl.zst). Install with:\n"
            "  pip install zstandard\n"
            "Or use a non-zstd corpus, e.g.\n"
            "  --dataset JeanKaddour/minipile"
        ) from exc

    # Make sure fsspec knows about zstd even on older stacks.
    try:
        import fsspec.compression as fsc

        if "zstd" not in getattr(fsc, "compr", {}):
            try:
                from fsspec.compression import register_compression
                from zstandard import ZstdDecompressor

                def _zstd_decompress(infile, mode="rb"):
                    dctx = ZstdDecompressor()
                    return dctx.stream_reader(infile)

                register_compression(
                    "zstd", _zstd_decompress, extensions=["zst", "zstd"]
                )
            except Exception:
                # Import alone is enough on recent fsspec/datasets.
                pass
    except Exception:
        pass


def build_pile_chunks(
    tokenizer,
    *,
    dataset: str = DEFAULT_PILE_DATASET,
    dataset_config: str | None = None,
    split: str = DEFAULT_PILE_SPLIT,
    seq_len: int,
    max_chunks: int,
    streaming: bool = True,
    text_field: str = "text",
) -> torch.Tensor:
    """Return ``[n_chunks, seq_len]`` token ids packed from The Pile.

    Streams documents until ``max_chunks * seq_len`` tokens are collected (or
    the stream ends). Same BOS/EOS packing convention as WikiText.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")
    if max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")

    # Default Pile mirror (and many others) use zstd shards.
    _ensure_zstd_support()

    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset, split=split, streaming=streaming)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    need = max_chunks * seq_len
    stream: list[int] = []

    try:
        rows_iter = iter(ds)
    except ValueError as exc:
        if "zstd" in str(exc).lower() or "compression" in str(exc).lower():
            raise ImportError(
                f"Failed to open {dataset} due to missing zstd support ({exc}). "
                "Install with: pip install zstandard"
            ) from exc
        raise

    for row in rows_iter:
        try:
            text = row.get(text_field) if isinstance(row, dict) else None
        except ValueError as exc:
            if "zstd" in str(exc).lower() or "compression" in str(exc).lower():
                raise ImportError(
                    f"Failed to read {dataset} due to missing zstd support ({exc}). "
                    "Install with: pip install zstandard\n"
                    "Or use: --dataset JeanKaddour/minipile"
                ) from exc
            raise
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
        raise ValueError("No Pile chunks produced")
    flat = torch.tensor(stream[: n * seq_len], dtype=torch.long)
    return flat.view(n, seq_len)


def dataset_tag(dataset: str, dataset_config: str | None, split: str) -> str:
    """Short filesystem-safe tag for output filenames."""
    parts = [dataset.replace("/", "_")]
    if dataset_config:
        parts.append(dataset_config.replace("/", "_"))
    parts.append(split)
    return "_".join(parts)
