"""CCNet next-token NLL under pairwise dark/middle residual swap.

Streams ``JorgeeGF/CCNet`` / ``train`` by default, packs text into
non-overlapping fixed-length token chunks, groups chunks into pairs
``(0,1), (2,3), …``, and measures causal LM NLL. For each layer window,
swap the dark or middle unembedding subspace **between the two paired chunks
at the same token positions** (same intervention as WikiText / LAMBADA
``swap_subspace_and_renormalize``), then re-score. Non-subspace residual stays
with the receiver; L2 is renormalized. Position 0 is left unchanged.
``--max-chunks`` and ``--batch-size`` must be even so every chunk has a partner.

Sweeps fixed-size windows over ``[--layer-start, --layer-end]`` with step
``--stride``. Default end is ``n_layers - 2``. ``--layers`` skips the sweep.
``none`` is scored once (unhooked). The dark subspace is the trailing
``--null-dim`` right-singular directions. The middle subspace is the band in
``mech_interp/wikitext_middle_nll_match`` (``*_nll_match_se_tol.csv``) whose
WikiText NLL matches the dark ablation. A missing band for this model, hook,
null rank, and layer window is an error.

Example:
  python scripts/ccnet/swap_dark_matter_nll.py Qwen3-8B \\
      --layer-start 20 --layer-end 24 --size 1 --stride 1 \\
      --null-dim 256 --ablation none dark --max-chunks 64
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "nq_swap"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablate_dark_matter_generate import (  # noqa: E402
    layers_tag,
    parse_layers,
    projector_for_mode,
)
from src.mech_interp.light_matter import (  # noqa: E402
    swap_subspace_and_renormalize,
    unembedding_right_basis,
)
from src.mech_interp.load_model import load_tl_model  # noqa: E402
from ccnet_utils import (  # noqa: E402
    DEFAULT_CCNET_DATASET,
    DEFAULT_CCNET_CONFIG,
    DEFAULT_CCNET_SPLIT,
    DEFAULT_CCNET_TEXT_FIELD,
    SWAP_MODES,
    build_ccnet_chunks,
    dataset_tag,
    layer_windows,
    mode_label,
)

load_dotenv()
login(token=os.getenv("HUGGING_FACE_API_KEY"))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mech_interp" / "ccnet_dark_ablation"
NLL_MATCH_DIR = PROJECT_ROOT / "mech_interp" / "wikitext_middle_nll_match"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_name", help="Model short name, e.g. Qwen3-8B")
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
        "--layer-start",
        type=int,
        default=0,
        help="Inclusive start of the layer interval to sweep (default: 0).",
    )
    p.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help=(
            "Inclusive end of the layer interval (default: n_layers - 2, "
            "so the final block is never ablated)."
        ),
    )
    p.add_argument(
        "--size",
        type=int,
        default=4,
        help="Number of consecutive layers per window (default: 4).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Step between window starts (default: --size, i.e. no overlap). "
            "Overlap = size - stride."
        ),
    )
    p.add_argument(
        "--layers",
        default=None,
        help=(
            "If set, only this layer set (e.g. 20-24); skips the window sweep. "
            "With this flag, --ablation all means dark+middle only (no none)."
        ),
    )
    p.add_argument(
        "--null-dim",
        type=int,
        nargs="+",
        default=[256],
        help="Rank(s) of the swapped subspace, e.g. --null-dim 64 256",
    )
    p.add_argument(
        "--ablation",
        nargs="+",
        choices=(*SWAP_MODES, "all"),
        default=["all"],
        help=(
            "One or more interventions, e.g. --ablation none dark. "
            "'all' = none once, then dark+middle per window."
        ),
    )
    p.add_argument(
        "--hook",
        choices=("resid_post", "resid_mid"),
        default="resid_post",
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
        help="Max number of CCNet chunks to score (even, default: 128).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Chunks per forward pass (must be even and >= 2; default: 4).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--nll-match-dir",
        type=Path,
        default=NLL_MATCH_DIR,
        help=(
            "Directory of *_nll_match_se_tol.csv bands from "
            "scripts/wikitext/match_middle_nll.py."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return p.parse_args()


def ablation_filename_tag(ablation_args: list[str], modes: list[str]) -> str:
    """Stable filename piece. ``all`` stays ``all``; otherwise ``none_dark_middle`` order."""
    if ablation_args == ["all"]:
        return "all"
    return "_".join(mode for mode in SWAP_MODES if mode in modes)


def load_nll_match_table(directory: Path, model_name: str, hook: str) -> pd.DataFrame:
    """Middle and dark bands from ``match_middle_nll.py`` for this model and hook."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No NLL-match directory {directory}. "
            "Run scripts/wikitext/match_middle_nll.py first."
        )
    paths = sorted(directory.glob("*_nll_match_se_tol.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No *_nll_match_se_tol.csv files in {directory}. "
            "Run scripts/wikitext/match_middle_nll.py first."
        )
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "model" not in frame.columns or "ablation" not in frame.columns:
            continue
        frame = frame.loc[frame["model"] == model_name].copy()
        if frame.empty:
            continue
        if "hook" in frame.columns:
            frame = frame.loc[frame["hook"] == hook]
        frame = frame.loc[frame["ablation"].isin(["dark_swap", "middle_swap"])]
        if frame.empty:
            continue
        frame["_source"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No NLL-match rows for model={model_name!r} hook={hook!r} in {directory}"
        )
    return pd.concat(frames, ignore_index=True)


def lookup_nll_match_band(
    table: pd.DataFrame,
    *,
    ablation: str,
    null_dim: int,
    layer_start: int,
    layer_end: int,
) -> tuple[int, int, int]:
    """Inclusive ``(band_start, band_end, rank)`` for one window."""
    mask = (
        (table["ablation"] == ablation)
        & (pd.to_numeric(table["null_dim"]) == null_dim)
        & (pd.to_numeric(table["layer_start"]) == layer_start)
        & (pd.to_numeric(table["layer_end"]) == layer_end)
    )
    rows = table.loc[mask]
    where = f"{ablation} null_dim={null_dim} layers={layer_start}-{layer_end}"
    if rows.empty:
        have = table.loc[
            table["ablation"] == ablation,
            ["null_dim", "layer_start", "layer_end", "_source"],
        ].drop_duplicates()
        raise FileNotFoundError(
            f"No NLL-match band for {where}. Rows on disk:\n{have.to_string(index=False)}"
        )
    bands = rows[["subspace_rank", "band_start", "band_end"]].drop_duplicates()
    if len(bands) != 1:
        sources = rows[
            ["_source", "subspace_rank", "band_start", "band_end"]
        ].drop_duplicates()
        raise ValueError(
            f"Conflicting NLL-match bands for {where}:\n{sources.to_string(index=False)}"
        )
    rank = int(bands["subspace_rank"].iloc[0])
    start = int(bands["band_start"].iloc[0])
    end = int(bands["band_end"].iloc[0])
    if rank <= 0 or end - start + 1 != rank:
        raise ValueError(
            f"NLL-match band for {where} has rank={rank} sv={start}..{end}"
        )
    return start, end, rank


def basis_from_vh(
    vh: torch.Tensor, start: int, rank: int, device: torch.device
) -> torch.Tensor:
    """Orthonormal basis ``[d_model, rank]`` from rows ``[start, start+rank)``."""
    return vh[start : start + rank].T.contiguous().to(device=device, dtype=torch.float32)


def pair_partner_index(chunk_idx: int) -> int:
    """Partner in fixed pairs (0,1), (2,3), …."""
    return chunk_idx ^ 1


def make_resid_pair_swap_hook(p_subspace: torch.Tensor) -> Callable:
    """Swap subspace between adjacent pair rows at matching token positions.

    Batch layout must be ``[a0, b0, a1, b1, …]``. Position 0 (BOS) is left
    unchanged; for every other position ``i``, the subspace of row ``2k``
    exchanges with the subspace of row ``2k+1``.
    """

    def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
        batch, seq_len, _ = resid.shape
        if batch % 2 != 0:
            raise ValueError(
                f"Pairwise DM swap needs an even batch size, got {batch}"
            )
        if seq_len <= 1:
            return resid
        perm = (
            torch.arange(batch, device=resid.device)
            .view(-1, 2)
            .flip(1)
            .reshape(-1)
        )
        patched = resid.clone()
        patched[:, 1:, :] = swap_subspace_and_renormalize(
            resid[:, 1:, :], p_subspace, perm
        )
        return patched

    return hook_fn


@torch.no_grad()
def batched_chunk_nll(
    tl_model,
    chunks: torch.Tensor,
    *,
    layers: list[int],
    hook_name: str,
    p_subspace: torch.Tensor | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token NLL and greedy-hit over next-token predictions.

    Returns ``nll, hit`` each of shape ``[n_chunks, seq_len - 1]``, where
    column ``j`` is the loss / hit for predicting ``chunks[:, j + 1]``.
    """
    n = chunks.shape[0]
    if n % 2 != 0:
        raise ValueError(f"Need an even number of chunks for pairing, got {n}")
    if batch_size < 2 or batch_size % 2 != 0:
        raise ValueError(
            f"--batch-size must be even and >= 2 for pairwise swap, got {batch_size}"
        )

    nll_chunks: list[torch.Tensor] = []
    hit_chunks: list[torch.Tensor] = []
    device = tl_model.cfg.device
    for start in tqdm(range(0, n, batch_size), desc="forward", leave=False):
        end = min(start + batch_size, n)
        if (end - start) % 2 != 0:
            end -= 1
        if end <= start:
            break
        batch = chunks[start:end].to(device)
        if p_subspace is None:
            logits = tl_model(batch)
        else:
            hook_fn = make_resid_pair_swap_hook(p_subspace)
            fwd_hooks = [
                (f"blocks.{layer}.{hook_name}", hook_fn) for layer in layers
            ]
            with tl_model.hooks(fwd_hooks=fwd_hooks):
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
    null_dims = list(dict.fromkeys(args.null_dim))
    if any(d <= 0 for d in null_dims):
        raise ValueError(f"--null-dim values must be positive, got {args.null_dim}")
    if args.seq_len < 2:
        raise ValueError(f"--seq-len must be >= 2, got {args.seq_len}")
    if args.max_chunks < 2 or args.max_chunks % 2 != 0:
        raise ValueError(
            f"--max-chunks must be even and >= 2 for pairing, got {args.max_chunks}"
        )
    if args.batch_size < 2 or args.batch_size % 2 != 0:
        raise ValueError(
            f"--batch-size must be even and >= 2 for pairing, got {args.batch_size}"
        )

    ablation_args = list(dict.fromkeys(args.ablation))
    if "all" in ablation_args:
        if ablation_args != ["all"]:
            raise ValueError("--ablation 'all' cannot be combined with other modes")
        modes = list(SWAP_MODES)
        if args.layers is not None:
            modes = [m for m in modes if m != "none"]
    else:
        modes = ablation_args
    hooked_modes = [mode for mode in modes if mode != "none"]

    tl_model = load_tl_model(args.model_name, args.device, args.attn_impl)
    n_layers = tl_model.cfg.n_layers
    if args.layers is not None:
        windows = [parse_layers(args.layers, n_layers)]
    else:
        layer_end = n_layers - 2 if args.layer_end is None else args.layer_end
        stride = args.size if args.stride is None else args.stride
        if args.layer_start < 0 or layer_end >= n_layers:
            raise ValueError(
                f"layer interval [{args.layer_start}, {layer_end}] out of range "
                f"for n_layers={n_layers} (valid indices 0..{n_layers - 1})"
            )
        windows = layer_windows(args.layer_start, layer_end, args.size, stride)
        args.layer_end = layer_end
        args.stride = stride

    hook_name = f"hook_{args.hook}"

    matched_bands: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    match_vh = None
    match_device = None
    if "middle" in modes:
        for layers in windows:
            if layers != list(range(layers[0], layers[-1] + 1)):
                raise ValueError(
                    f"{layers_tag(layers)} is not a contiguous window; "
                    "NLL-match bands are keyed by layer_start and layer_end."
                )
        match_table = load_nll_match_table(
            args.nll_match_dir, args.model_name, args.hook
        )
        match_vh = unembedding_right_basis(tl_model.W_U.detach())
        match_device = tl_model.W_U.device
        n_sv = match_vh.shape[0]
        for null_dim in null_dims:
            if null_dim > n_sv:
                raise ValueError(f"--null-dim {null_dim} exceeds n_sv={n_sv}")
            dark_start, dark_end = n_sv - null_dim, n_sv - 1
            for layers in windows:
                file_dark = lookup_nll_match_band(
                    match_table,
                    ablation="dark_swap",
                    null_dim=null_dim,
                    layer_start=layers[0],
                    layer_end=layers[-1],
                )
                if file_dark != (dark_start, dark_end, null_dim):
                    raise ValueError(
                        f"{layers_tag(layers)} null_dim={null_dim}: NLL-match dark band "
                        f"sv={file_dark[0]}..{file_dark[1]} rank={file_dark[2]} "
                        f"does not match this model's trailing band "
                        f"sv={dark_start}..{dark_end}"
                    )
                matched_bands[(null_dim, layers[0], layers[-1])] = lookup_nll_match_band(
                    match_table,
                    ablation="middle_swap",
                    null_dim=null_dim,
                    layer_start=layers[0],
                    layer_end=layers[-1],
                )
        print(
            f"NLL-match bands: {args.nll_match_dir}  "
            f"windows={len(matched_bands)}"
        )

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
    if chunks.shape[0] % 2 != 0:
        chunks = chunks[:-1]
    if chunks.shape[0] < 2:
        raise ValueError("Need at least 2 CCNet chunks to form a swap pair")
    ds_tag = dataset_tag(args.dataset, args.dataset_config, args.split)
    print(
        f"{args.model_name}  {ds_tag}  "
        f"chunks={chunks.shape[0]} (paired)  seq_len={args.seq_len}  "
        f"windows={len(windows)}  null_dims={null_dims}"
    )
    print("Layer windows:", ", ".join(layers_tag(w) for w in windows))

    rows: list[dict] = []

    def record(
        *,
        ablation: str,
        null_dim: int,
        layers: list[int],
        nll: torch.Tensor,
        hit: torch.Tensor,
        subspace_rank: int | None = None,
        band_start: int | None = None,
        band_end: int | None = None,
    ) -> None:
        """Append one CSV row per (chunk, token) prediction."""
        label = mode_label(ablation) if ablation != "none" else "none"
        layer_start = "" if not layers else layers[0]
        layer_end = "" if not layers else layers[-1]
        tag = "" if not layers else layers_tag(layers)
        n_chunks, n_pos = nll.shape
        chunk_means = nll.mean(dim=-1)
        chunk_accs = hit.float().mean(dim=-1)
        for i in range(n_chunks):
            nll_mean = float(chunk_means[i])
            partner = pair_partner_index(i)
            for j in range(n_pos):
                rows.append(
                    {
                        "chunk_idx": i,
                        "partner_chunk_idx": partner,
                        "pair_idx": i // 2,
                        "token_pos": j,
                        # Absolute index of the predicted token in the chunk.
                        "pred_pos": j + 1,
                        "ablation": label,
                        "null_dim": null_dim,
                        "subspace_rank": "" if subspace_rank is None else subspace_rank,
                        "band_start": "" if band_start is None else band_start,
                        "band_end": "" if band_end is None else band_end,
                        "layers": tag,
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "hook": args.hook,
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
        band = ""
        if subspace_rank is not None:
            band = f"  rank={subspace_rank} sv={band_start}..{band_end}"
        print(
            f"  {label:12s}  {tag or '—':12s}  nd={null_dim:<3d}{band}  "
            f"NLL={float(nll.mean()):.3f}  acc={float(hit.float().mean()):.3f}"
        )

    if "none" in modes:
        nll, hit = batched_chunk_nll(
            tl_model,
            chunks,
            layers=[],
            hook_name=hook_name,
            p_subspace=None,
            batch_size=args.batch_size,
        )
        record(ablation="none", null_dim=0, layers=[], nll=nll, hit=hit)

    for null_dim in null_dims:
        projectors = {
            mode: projector_for_mode(mode, tl_model, null_dim)
            for mode in hooked_modes
            if mode != "middle"
        }
        for layers in windows:
            for mode in hooked_modes:
                if mode == "middle":
                    start, end, rank = matched_bands[(null_dim, layers[0], layers[-1])]
                    assert match_vh is not None and match_device is not None
                    p_subspace = basis_from_vh(match_vh, start, rank, match_device)
                    band_kwargs = {
                        "subspace_rank": rank,
                        "band_start": start,
                        "band_end": end,
                    }
                else:
                    p_subspace = projectors[mode]
                    band_kwargs = {}
                nll, hit = batched_chunk_nll(
                    tl_model,
                    chunks,
                    layers=layers,
                    hook_name=hook_name,
                    p_subspace=p_subspace,
                    batch_size=args.batch_size,
                )
                record(
                    ablation=mode,
                    null_dim=null_dim,
                    layers=layers,
                    nll=nll,
                    hit=hit,
                    **band_kwargs,
                )

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")
    if len(null_dims) == 1:
        nd_tag = f"nd{null_dims[0]}"
    else:
        nd_tag = f"nd{null_dims[0]}-{null_dims[-1]}"
    if args.layers is not None:
        stem = (
            f"{model_tag}_{ds_tag}_"
            f"{layers_tag(windows[0])}_{nd_tag}_"
            f"s{args.seq_len}_n{chunks.shape[0]}"
        )
    else:
        stem = (
            f"{model_tag}_{ds_tag}_"
            f"L{args.layer_start}-{args.layer_end}_"
            f"sz{args.size}_st{args.stride}_{nd_tag}_"
            f"s{args.seq_len}_n{chunks.shape[0]}"
        )
    name_parts = [stem]
    if "middle" in modes:
        name_parts.append("nllmatch")
    name_parts.append(ablation_filename_tag(ablation_args, modes))
    out_csv = args.output_dir / ("_".join(name_parts) + "_swap.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    summary = (
        df.groupby(["ablation", "null_dim", "layer_start"], dropna=False)["nll"]
        .mean()
        .unstack("ablation")
    )
    print(summary)


if __name__ == "__main__":
    main()
