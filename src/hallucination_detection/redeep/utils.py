import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from scipy.stats import rankdata
from tqdm import trange

from src.llm import LLM


def iqr_outlier_count(points):
    if len(points) == 0:
        return 0
    q1 = np.percentile(points, 25)
    q3 = np.percentile(points, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(np.sum((points < lower) | (points > upper)))


@torch.no_grad()
def copying_head_scores(
    model,
    num_chunks: int = 8,
    chunk_size: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    """Per-head copying scores from trace + Gershgorin outlier counts.

    For head ``h``, the OV circuit is ``W_OV = W_O[:, s:e] @ W_V[s:e, :]``
    (HuggingFace ``Linear`` layout). Copying is scored on chunks of
    ``M = W_U @ W_OV @ W_E.T``. Rank by few IQR outliers (ascending) and
    ``|tr(M)|`` (descending), matching Appendix B of ReDeEP.
    """
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_value_heads = model.config.num_key_value_heads

    Wu = model.lm_head.weight
    We = model.model.embed_tokens.weight

    if device is None:
        device = Wu.device
    if dtype is None:
        dtype = Wu.dtype

    Wu = Wu.to(device=device, dtype=dtype)
    We = We.to(device=device, dtype=dtype)

    V, _ = Wu.shape
    if chunk_size is None:
        chunk_size = max(1, V // num_chunks)

    abs_traces, outlier_counts = [], []

    for layer_idx in trange(num_layers, desc="Iterating layers", total=num_layers):
        layer = model.model.layers[layer_idx]
        Wo = layer.self_attn.o_proj.weight.to(device=device, dtype=dtype)
        Wv = layer.self_attn.v_proj.weight.to(device=device, dtype=dtype)
        Wv = Wv.repeat_interleave(num_heads // num_value_heads, dim=0)

        for head_idx in trange(num_heads, desc=f"Layer {layer_idx}: Iterating heads", total=num_heads, leave=False):
            s = head_idx * head_dim
            e = (head_idx + 1) * head_dim
            Wov = Wo[:, s:e].matmul(Wv[s:e, :])

            centers_list = []
            row_abs_sums_list = []
            for start in range(0, V, chunk_size):
                end = min(V, start + chunk_size)
                Wu_chunk = Wu[start:end]
                M_chunk = Wu_chunk.matmul(Wov).matmul(We.t())
                rows = end - start
                local_rows = torch.arange(rows, device=device)
                global_cols = torch.arange(start, end, device=device)
                diag_chunk = M_chunk[local_rows, global_cols]
                row_abs_sums = torch.sum(torch.abs(M_chunk), dim=1)
                # bfloat16 cannot be converted with Tensor.numpy()
                centers_list.append(diag_chunk.float().cpu().numpy())
                row_abs_sums_list.append(row_abs_sums.float().cpu().numpy())
                del M_chunk

            centers = np.concatenate(centers_list)
            abs_traces.append(abs(float(sum(centers))))
            row_abs_sums = np.concatenate(row_abs_sums_list)
            radii = row_abs_sums - np.abs(centers)
            boundary_points = np.concatenate([centers + radii, centers - radii])
            outlier_counts.append(iqr_outlier_count(boundary_points))

    abs_traces = np.array(abs_traces)
    outlier_counts = np.array(outlier_counts)
    rank_outliers = rankdata(outlier_counts, method="average")
    rank_trace = rankdata(-abs_traces, method="average")
    return rank_outliers + rank_trace


AUTHOR_COPY_HEADS_DIR = Path(__file__).resolve().parent / "author_copy_heads"
AUTHOR_COPY_HEAD_FILES = {
    "Llama-2-7b-chat-hf": "Llama-2-7b-chat-hf.json",
    "Llama-2-13b-chat-hf": "Llama-2-13b-chat-hf.json",
    "Llama-3-8B-Instruct": "Meta-Llama-3-8B-Instruct.json",
}

# ReDeEP(Token) on RAGTruth: how many ranked copy heads and FFN layers to keep.
# Appendix, Implementation Details.
REDEEP_TOKEN_RAGTRUTH: Dict[str, Dict[str, int]] = {
    "Llama-2-7b-chat-hf": {"top_n": 1, "top_k": 10},
    "Llama-2-13b-chat-hf": {"top_n": 2, "top_k": 17},
    "Llama-3-8B-Instruct": {"top_n": 3, "top_k": 30},
}


def _load_head_list(path: Path) -> List[Tuple[int, int]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(int(layer), int(head)) for layer, head in raw]


def redeep_token_hparams(model_name: str) -> dict[str, int]:
    """``top_n`` copy heads and ``top_k`` FFN layers for ReDeEP(Token) on RAGTruth."""
    key = str(model_name).split("/")[-1]
    if key not in REDEEP_TOKEN_RAGTRUTH:
        known = ", ".join(sorted(set(REDEEP_TOKEN_RAGTRUTH)))
        raise KeyError(
            f"No ReDeEP(Token) RAGTruth hyperparameters for {model_name!r}. Known models: {known}"
        )
    return dict(REDEEP_TOKEN_RAGTRUTH[key])


def author_copy_heads_path(model_name: str) -> Optional[Path]:
    filename = AUTHOR_COPY_HEAD_FILES.get(model_name)
    if filename is None:
        return None
    path = AUTHOR_COPY_HEADS_DIR / filename
    return path if path.exists() else None


def find_copy_heads(
    model: LLM,
    cache_path: Optional[str] = None,
) -> List[Tuple[int, int]]:
    if cache_path is not None:
        path = Path(cache_path)
        if path.exists():
            heads = _load_head_list(path)
            logger.info(f"Loaded {len(heads)} copy heads from {path}")
            return heads

    scores = copying_head_scores(model.model)
    n_keep = model.model.config.num_attention_heads
    indices_1d = np.argsort(scores)[:n_keep]
    num_layers = len(model.model.model.layers)
    num_heads = model.model.config.num_attention_heads
    layer_idx, head_idx = np.unravel_index(indices_1d, shape=(num_layers, num_heads))
    heads = [(int(layer), int(head)) for layer, head in zip(layer_idx, head_idx)]

    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(heads, f)
        logger.info(f"Saved {len(heads)} copy heads to {path}")
    return heads


def resolve_copy_heads(
    model: LLM,
    copy_heads: Optional[Sequence[Sequence[int]]],
    cache_path: Optional[str] = None,
) -> List[Tuple[int, int]]:
    if copy_heads:
        return [(int(layer), int(head)) for layer, head in copy_heads]
    author_path = author_copy_heads_path(model.model_name)
    if author_path is not None:
        heads = _load_head_list(author_path)
        logger.info(f"Using authors' copy heads ({len(heads)}) from {author_path}")
        return heads
    logger.warning(
        f"No authors' copy heads for {model.model_name}; computing from OV-circuit scores"
    )
    return find_copy_heads(model, cache_path=cache_path)
