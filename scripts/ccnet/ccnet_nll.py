"""Unablated CCNet next-token NLL (baseline, no residual swap).

Streams ``JorgeeGF/CCNet`` / ``train`` (default) with the same packing
as ``swap_dark_matter_nll.py`` and scores causal LM NLL / greedy accuracy with
no interventions.

Example:
  python scripts/ccnet/ccnet_nll.py Qwen3.5-2B \\
      --seq-len 256 --max-chunks 128 --batch-size 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mech_interp.load_model import load_tl_model  # noqa: E402
from ccnet_utils import (  # noqa: E402
    DEFAULT_CCNET_DATASET,
    DEFAULT_CCNET_CONFIG,
    DEFAULT_CCNET_SPLIT,
    DEFAULT_CCNET_TEXT_FIELD,
    build_ccnet_chunks,
    dataset_tag,
)

load_dotenv()
login(token=os.getenv("HUGGING_FACE_API_KEY"))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mech_interp" / "ccnet_nll"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "model_name",
        nargs="?",
        default="Qwen3.5-2B",
        help="Model short name (default: Qwen3.5-2B).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument(
        "--dataset",
        default=DEFAULT_CCNET_DATASET,
        help=f"HF datasets name (default: {DEFAULT_CCNET_DATASET}).",
    )
    p.add_argument(
        "--dataset-config",
        default=None,
        help="Optional HF config name (CCNet usually needs none).",
    )
    p.add_argument(
        "--split",
        default=DEFAULT_CCNET_SPLIT,
        help=f"Dataset split (default: {DEFAULT_CCNET_SPLIT}).",
    )
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="Load the full split into memory instead of streaming.",
    )
    p.add_argument(
        "--text-field",
        default=DEFAULT_CCNET_TEXT_FIELD,
        help=f"Column containing document text (default: {DEFAULT_CCNET_TEXT_FIELD}).",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Token chunk length (default: 256).",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=128,
        help="Max number of CCNet chunks to score (default: 128).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Chunks per forward pass (default: 4).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


@torch.no_grad()
def batched_chunk_nll(
    tl_model,
    chunks: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token NLL and greedy-hit over next-token predictions.

    Returns ``nll, hit`` each of shape ``[n_chunks, seq_len - 1]``.
    """
    nll_chunks: list[torch.Tensor] = []
    hit_chunks: list[torch.Tensor] = []
    n = chunks.shape[0]
    device = tl_model.cfg.device
    for start in tqdm(range(0, n, batch_size), desc="forward", leave=False):
        batch = chunks[start : start + batch_size].to(device)
        logits = tl_model(batch)
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = batch[:, 1:]
        nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        hit = logits[:, :-1, :].argmax(dim=-1) == targets
        nll_chunks.append(nll.detach().cpu())
        hit_chunks.append(hit.detach().cpu())
        del logits, logp, nll, hit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(nll_chunks, dim=0), torch.cat(hit_chunks, dim=0)


def main() -> None:
    args = parse_args()
    if args.dataset_config == "":
        args.dataset_config = None
    if args.seq_len < 2:
        raise ValueError(f"--seq-len must be >= 2, got {args.seq_len}")
    if args.max_chunks < 1:
        raise ValueError(f"--max-chunks must be >= 1, got {args.max_chunks}")

    torch.manual_seed(args.seed)
    tl_model = load_tl_model(args.model_name, args.device, args.attn_impl)
    tokenizer = tl_model.tokenizer
    if tokenizer is None:
        raise RuntimeError("Model has no tokenizer; cannot pack CCNet chunks")

    chunks = build_ccnet_chunks(
        tokenizer,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        seq_len=args.seq_len,
        max_chunks=args.max_chunks,
        streaming=not args.no_streaming,
        text_field=args.text_field,
    )
    ds_tag = dataset_tag(args.dataset, args.dataset_config, args.split)
    print(
        f"{args.model_name}  {ds_tag}  "
        f"chunks={chunks.shape[0]}  seq_len={args.seq_len}  (unablated)"
    )

    nll, hit = batched_chunk_nll(tl_model, chunks, batch_size=args.batch_size)
    mean_nll = float(nll.mean())
    mean_acc = float(hit.float().mean())
    print(f"  NLL={mean_nll:.4f}  acc={mean_acc:.4f}")

    rows: list[dict] = []
    n_chunks, n_pos = nll.shape
    chunk_means = nll.mean(dim=-1)
    chunk_accs = hit.float().mean(dim=-1)
    for i in range(n_chunks):
        nll_mean = float(chunk_means[i])
        for j in range(n_pos):
            rows.append(
                {
                    "chunk_idx": i,
                    "token_pos": j,
                    "pred_pos": j + 1,
                    "ablation": "none",
                    "nll": float(nll[i, j]),
                    "hit": int(hit[i, j].item()),
                    "nll_mean": nll_mean,
                    "acc": float(chunk_accs[i]),
                    "seq_len": args.seq_len,
                    "dataset": args.dataset,
                    "dataset_config": args.dataset_config or "",
                    "split": args.split,
                }
            )

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")
    out_csv = (
        args.output_dir
        / (
            f"{model_tag}_{ds_tag}_"
            f"s{args.seq_len}_n{chunks.shape[0]}_nll.csv"
        )
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    print(
        f"chunk NLL: mean={float(chunk_means.mean()):.4f}  "
        f"std={float(chunk_means.std()):.4f}  "
        f"acc={mean_acc:.4f}"
    )


if __name__ == "__main__":
    main()
