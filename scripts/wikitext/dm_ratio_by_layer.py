"""Mean dark-matter ratio of the residual after every layer on WikiText.

Dark matter is ``||P_null h|| / ||h||`` at ``resid_post``. ``P_null`` is the
trailing right-singular subspace of the unembedding. The null rank defaults to
5% of the residual width (4096 → 205, 5120 → 256).

Text is ``Salesforce/wikitext`` / ``wikitext-2-raw-v1`` / ``test``, packed into
non-overlapping chunks the same way as the WikiText NLL scripts. The reported
value is the mean over every token in those chunks.

Example:
  python scripts/wikitext/dm_ratio_by_layer.py --model-name Qwen3-8B --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.mech_interp.light_matter import (  # noqa: E402
    light_matter_ratio,
    unembedding_null_basis,
)
from src.mech_interp.load_model import load_tl_model  # noqa: E402
from wikitext_utils import build_wikitext_chunks  # noqa: E402

load_dotenv()
token = os.getenv("HUGGING_FACE_API_KEY")
if token:
    login(token=token)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mech_interp" / "wikitext_dm_ratio"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--null-dim",
        type=int,
        default=None,
        help="Null rank. Default: 5%% of d_model, rounded.",
    )
    p.add_argument("--hook", choices=("resid_post", "resid_mid"), default="resid_post")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.seq_len < 1 or args.batch_size < 1:
        raise ValueError("--num-samples, --seq-len, and --batch-size must be positive")

    tl_model = load_tl_model(args.model_name, args.device, args.attn_impl)
    n_layers = tl_model.cfg.n_layers
    null_dim = args.null_dim
    if null_dim is None:
        null_dim = max(1, int(round(0.05 * tl_model.cfg.d_model)))
    basis = unembedding_null_basis(tl_model.W_U.detach(), null_dim=null_dim)

    chunks = build_wikitext_chunks(
        tl_model.tokenizer,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        seq_len=args.seq_len,
        max_chunks=args.num_samples,
    ).to(args.device)
    n_samples = chunks.shape[0]
    hook = f"hook_{args.hook}"

    def names_filter(name: str) -> bool:
        return name.endswith(f".{hook}")

    dm_sum = torch.zeros(n_layers, dtype=torch.float64)
    n_tokens = n_samples * args.seq_len
    for start in tqdm(range(0, n_samples, args.batch_size), desc="wikitext"):
        batch = chunks[start : start + args.batch_size]
        with torch.no_grad():
            _, cache = tl_model.run_with_cache(batch, names_filter=names_filter)
        for layer in range(n_layers):
            hidden = cache[f"blocks.{layer}.{hook}"]
            dm = 1.0 - light_matter_ratio(hidden, basis)
            dm_sum[layer] += dm.detach().double().sum().cpu()
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_dm = (dm_sum / n_tokens).numpy()
    df = pd.DataFrame(
        {
            "layer": range(n_layers),
            "mean_dm": mean_dm,
            "n_tokens": n_tokens,
            "n_samples": n_samples,
            "seq_len": args.seq_len,
            "null_dim": null_dim,
            "hook": args.hook,
            "model": args.model_name,
            "dataset": args.dataset_config,
            "split": args.split,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{args.model_name}_{args.dataset_config}_{args.split}"
        f"_nd{null_dim}_s{args.seq_len}_n{n_samples}"
    )
    out = args.output_dir / f"{tag}_dm_by_layer.csv"
    df.to_csv(out, index=False)
    peak = int(df["mean_dm"].idxmax())
    print(
        f"{n_samples} chunks × {args.seq_len} tokens, null_dim={null_dim}, "
        f"mean DM {df['mean_dm'].mean():.4f}, "
        f"peak layer {peak} ({df.loc[peak, 'mean_dm']:.4f})"
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
