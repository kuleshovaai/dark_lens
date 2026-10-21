"""Induction copying loss after swapping residual dark / middle components.

Same sequences and metrics as ``ablate_induction_loss.py``:

    [BOS, x_1 .. x_T, x_1 .. x_T]

Instead of *deleting* a rank-``null_dim`` subspace, this replaces that
component at every non-BOS position with the same-position component from
another synthetic sample in the batch (circular shift). Light / non-subspace
directions stay with the original sequence. After the swap the residual is
renormalized to the receiver's original L2, matching the delete ablation.

Sweeps fixed-size sliding windows over ``[--layer-start, --layer-end]`` with
step ``--window-stride`` (default: ``--size``, i.e. no overlap). Default end
is ``n_layers - 2`` so the final block is never swapped. ``--layers`` skips
the sweep and runs a single layer set. ``--null-dim`` selects the dark
subspace rank (default: sweep 64, 128, 256). The middle band is not chosen
here: for each window it is the band in
``mech_interp/wikitext_middle_nll_match/`` whose WikiText NLL damage matches
the dark ablation. ``none`` runs once. The output filename includes the
ablation modes that were scored, so a middle-only run does not replace a
file that also contains ``none`` and ``dark``.

``--batch-size`` must be at least 2 so every sequence has a donor.

Example:
  # Non-overlapping 4-layer windows over [0, 34]
  python scripts/all/swap_induction_loss.py Qwen3-8B \\
      --layer-start 0 --layer-end 34 --size 4 --window-stride 4 --stride 8

  # Single window
  python scripts/all/swap_induction_loss.py Qwen3-8B \\
      --layers 24-31 --stride 8 --ablation none dark
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "all"))

from ablate_induction_loss import (  # noqa: E402
    NULL_DIMS,
    build_induction_tokens,
    layers_tag,
    parse_layers,
    repeat_offsets,
    summarize_offsets,
)
from src.mech_interp.light_matter import (  # noqa: E402
    swap_subspace_and_renormalize,
    unembedding_right_basis,
)
from src.mech_interp.load_model import load_tl_model  # noqa: E402

load_dotenv()
login(token=os.getenv("HUGGING_FACE_API_KEY"))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mech_interp" / "induction_ablation"
NLL_MATCH_DIR = PROJECT_ROOT / "mech_interp" / "wikitext_middle_nll_match"
SWAP_MODES = ("none", "dark", "middle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_name", help="Model short name, e.g. Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="eager")
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
            "so the final block is never swapped)."
        ),
    )
    p.add_argument(
        "--size",
        type=int,
        default=4,
        help="Number of consecutive layers per window (default: 4).",
    )
    p.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help=(
            "Step between window starts (default: --size, i.e. no overlap). "
            "Overlap = size - window-stride."
        ),
    )
    p.add_argument(
        "--layers",
        default=None,
        help=(
            "If set, only this layer set (e.g. 24-31); skips the window sweep. "
            "With this flag, --ablation all means dark+middle only (no none)."
        ),
    )
    p.add_argument(
        "--null-dim",
        type=int,
        default=None,
        help=(
            "Rank of the dark subspace. The middle band for each window is read "
            "from the WikiText NLL-match CSVs. If omitted, "
            f"sweeps {', '.join(str(d) for d in NULL_DIMS)}."
        ),
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
    p.add_argument("--hook", choices=("resid_post", "resid_mid"), default="resid_post")
    p.add_argument("--num-seqs", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=64, help="Length T of one copy.")
    p.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Evaluate every k-th token of both the prefix and the repeat.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--shift",
        type=int,
        default=1,
        help="Batch-axis circular shift for the donor sample (must be nonzero mod B).",
    )
    p.add_argument("--vocab-low", type=int, default=100)
    p.add_argument("--vocab-high", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    p.add_argument(
        "--nll-match-dir",
        type=Path,
        default=NLL_MATCH_DIR,
        help=(
            "Directory of match_middle_nll.py CSVs. Middle swaps use the "
            "middle_swap band for the same model, hook, null dim, and layers."
        ),
    )
    return p.parse_args()


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
        raise ValueError(f"--window-stride must be positive, got {stride}")
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


def ablation_filename_tag(modes: list[str]) -> str:
    """Stable filename piece, in ``none_dark_middle`` order."""
    return "_".join(mode for mode in SWAP_MODES if mode in modes)


def mode_label(mode: str) -> str:
    return "none" if mode == "none" else f"{mode}_swap"


def apply_skip_bos_swap(
    resid: torch.Tensor,
    basis: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    batch = resid.shape[0]
    if batch < 2:
        raise ValueError("DM swap needs batch size >= 2")
    if shift % batch == 0:
        raise ValueError(
            f"--shift={shift} is 0 mod batch={batch}; every sample would donate to itself"
        )
    if resid.shape[1] <= 1:
        return resid
    perm = (torch.arange(batch, device=resid.device) + shift) % batch
    patched = resid.clone()
    patched[:, 1:, :] = swap_subspace_and_renormalize(resid[:, 1:, :], basis, perm)
    return patched


def make_resid_hook(*, basis: torch.Tensor | None, shift: int):
    def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
        if basis is None:
            return resid
        return apply_skip_bos_swap(resid, basis, shift)

    return hook_fn


def load_nll_match_table(directory: Path, model_name: str, hook: str) -> pd.DataFrame:
    """Middle and dark bands from ``match_middle_nll.py`` for this model and hook."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No NLL-match directory {directory}. "
            "Run scripts/wikitext/match_middle_nll.py first."
        )
    paths = sorted(directory.glob("*_nll_match_tol0.01.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No *_nll_match.csv files in {directory}. "
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
    where = (
        f"{ablation} null_dim={null_dim} layers={layer_start}-{layer_end}"
    )
    if rows.empty:
        have = table.loc[
            table["ablation"] == ablation, ["null_dim", "layer_start", "layer_end", "_source"]
        ].drop_duplicates()
        raise FileNotFoundError(
            f"No NLL-match band for {where}. Rows on disk:\n{have.to_string(index=False)}"
        )
    bands = rows[["subspace_rank", "band_start", "band_end"]].drop_duplicates()
    if len(bands) != 1:
        sources = rows[["_source", "subspace_rank", "band_start", "band_end"]].drop_duplicates()
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


def basis_from_vh(vh: torch.Tensor, start: int, rank: int, device: torch.device) -> torch.Tensor:
    """Orthonormal basis ``[d_model, rank]`` from rows ``[start, start+rank)``."""
    return vh[start : start + rank].T.contiguous().to(device=device, dtype=torch.float32)


@torch.no_grad()
def batched_nll(
    tl_model,
    tokens: torch.Tensor,
    *,
    layers: list[int],
    hook_name: str,
    basis: torch.Tensor | None,
    shift: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-position NLL and greedy-hit on targets ``tokens[:, 1:]``."""
    if batch_size < 2:
        raise ValueError(f"--batch-size must be >= 2 for a swap, got {batch_size}")
    nll_chunks: list[torch.Tensor] = []
    hit_chunks: list[torch.Tensor] = []
    n = tokens.shape[0]
    if n < 2:
        raise ValueError("--num-seqs must be >= 2 for a swap")
    hook_fn = make_resid_hook(basis=basis, shift=shift)
    fwd_hooks = (
        [(f"blocks.{layer}.{hook_name}", hook_fn) for layer in layers]
        if basis is not None
        else []
    )
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        if end - start == 1:
            spans[-1] = (spans[-1][0], n)
            break
        spans.append((start, end))
        start = end
    for start, end in tqdm(spans, desc="forward", leave=False):
        batch = tokens[start:end]
        if batch.shape[0] < 2:
            raise ValueError(f"Swap batch [{start}:{end}] has size {batch.shape[0]}")
        if fwd_hooks:
            with tl_model.hooks(fwd_hooks=fwd_hooks):
                logits = tl_model(batch)
        else:
            logits = tl_model(batch)
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = batch[:, 1:]
        nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        hit = logits[:, :-1, :].argmax(dim=-1) == targets
        nll_chunks.append(nll.detach().cpu())
        hit_chunks.append(hit.detach().cpu())
        del logits, logp, nll
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(nll_chunks, dim=0), torch.cat(hit_chunks, dim=0)


def annotate_layers(recs: list[dict], layers: list[int]) -> list[dict]:
    layer_start = "" if not layers else layers[0]
    layer_end = "" if not layers else layers[-1]
    tag = "" if not layers else layers_tag(layers)
    for r in recs:
        r["layers"] = tag
        r["layer_start"] = layer_start
        r["layer_end"] = layer_end
    return recs


def main() -> None:
    args = parse_args()
    if args.shift == 0:
        raise ValueError("--shift must be nonzero")

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
    offsets = repeat_offsets(args.seq_len, args.stride)

    tl_model = load_tl_model(args.model_name, args.device, args.attn_impl)
    n_layers = tl_model.cfg.n_layers
    if args.layers is not None:
        windows = [parse_layers(args.layers, n_layers)]
    else:
        layer_end = n_layers - 2 if args.layer_end is None else args.layer_end
        window_stride = args.size if args.window_stride is None else args.window_stride
        if args.layer_start < 0 or layer_end >= n_layers:
            raise ValueError(
                f"layer interval [{args.layer_start}, {layer_end}] out of range "
                f"for n_layers={n_layers} (valid indices 0..{n_layers - 1})"
            )
        windows = layer_windows(
            args.layer_start, layer_end, args.size, window_stride
        )
        args.layer_end = layer_end
        args.window_stride = window_stride

    hook_name = f"hook_{args.hook}"
    bos_id = tl_model.tokenizer.bos_token_id if tl_model.tokenizer is not None else None
    tokens = build_induction_tokens(
        bos_id=bos_id,
        num_seqs=args.num_seqs,
        seq_len=args.seq_len,
        vocab_low=args.vocab_low,
        vocab_high=args.vocab_high,
        seed=args.seed,
    ).to(device=tl_model.cfg.device)

    null_dims = list(NULL_DIMS) if args.null_dim is None else [args.null_dim]
    if any(d <= 0 for d in null_dims):
        raise ValueError(f"--null-dim must be positive, got {null_dims}")

    print(
        f"{args.model_name}  T={args.seq_len}  eval_stride={args.stride}  "
        f"offsets={offsets}  windows={len(windows)}  null_dims={null_dims}  "
        f"shift={args.shift}"
    )
    print("Layer windows:", ", ".join(layers_tag(w) for w in windows))

    rows: list[dict] = []

    if "none" in modes:
        nll, hit = batched_nll(
            tl_model,
            tokens,
            layers=[],
            hook_name=hook_name,
            basis=None,
            shift=args.shift,
            batch_size=args.batch_size,
        )
        recs = annotate_layers(
            summarize_offsets(
                nll,
                hit,
                ablation=mode_label("none"),
                null_dim=0,
                seq_len=args.seq_len,
                offsets=offsets,
            ),
            [],
        )
        for rec in recs:
            rec["subspace_rank"] = 0
        rows.extend(recs)
        mean_row = next(r for r in recs if r["kind"] == "repeat_mean")
        prefix_mean = next(r for r in recs if r["kind"] == "prefix_mean")
        print(
            f"  {'none':12s}  nd=0    prefix mean NLL={prefix_mean['nll']:.3f}  "
            f"repeat mean NLL={mean_row['nll']:.3f}  acc={mean_row['acc']:.3f}"
        )

    if hooked_modes:
        vh = unembedding_right_basis(tl_model.W_U.detach())
        n_sv = vh.shape[0]
        device = tl_model.W_U.device
    match_table = None
    if "middle" in hooked_modes:
        match_table = load_nll_match_table(args.nll_match_dir, args.model_name, args.hook)
        print(f"NLL-match bands: {args.nll_match_dir}  rows={len(match_table)}")
    for null_dim in null_dims:
        if not hooked_modes:
            break
        if null_dim > n_sv:
            raise ValueError(f"--null-dim {null_dim} exceeds n_sv={n_sv}")
        dark_start = n_sv - null_dim
        dark_end = n_sv - 1
        dark_basis = basis_from_vh(vh, dark_start, null_dim, device)
        for layers in windows:
            tag = layers_tag(layers)
            mid_basis = None
            mid_rank = None
            mid_start = None
            mid_end = None
            if "middle" in hooked_modes:
                assert match_table is not None
                file_dark = lookup_nll_match_band(
                    match_table,
                    ablation="dark_swap",
                    null_dim=null_dim,
                    layer_start=layers[0],
                    layer_end=layers[-1],
                )
                if file_dark != (dark_start, dark_end, null_dim):
                    raise ValueError(
                        f"{tag} null_dim={null_dim}: NLL-match dark band "
                        f"sv={file_dark[0]}..{file_dark[1]} rank={file_dark[2]} "
                        f"does not match this model's trailing band "
                        f"sv={dark_start}..{dark_end}"
                    )
                mid_start, mid_end, mid_rank = lookup_nll_match_band(
                    match_table,
                    ablation="middle_swap",
                    null_dim=null_dim,
                    layer_start=layers[0],
                    layer_end=layers[-1],
                )
                mid_basis = basis_from_vh(vh, mid_start, mid_rank, device)
            bases = {"dark": dark_basis, "middle": mid_basis}
            ranks = {"dark": null_dim, "middle": mid_rank}
            spans = {
                "dark": (dark_start, dark_end),
                "middle": (mid_start, mid_end),
            }
            for mode in hooked_modes:
                nll, hit = batched_nll(
                    tl_model,
                    tokens,
                    layers=layers,
                    hook_name=hook_name,
                    basis=bases[mode],
                    shift=args.shift,
                    batch_size=args.batch_size,
                )
                label = mode_label(mode)
                recs = annotate_layers(
                    summarize_offsets(
                        nll,
                        hit,
                        ablation=label,
                        null_dim=null_dim,
                        seq_len=args.seq_len,
                        offsets=offsets,
                    ),
                    layers,
                )
                for rec in recs:
                    rec["subspace_rank"] = ranks[mode]
                    rec["band_start"] = spans[mode][0]
                    rec["band_end"] = spans[mode][1]
                rows.extend(recs)
                stride_rows = [r for r in recs if r["kind"] == "stride"]
                prefix_rows = [r for r in recs if r["kind"] == "prefix_stride"]
                mean_row = next(r for r in recs if r["kind"] == "repeat_mean")
                prefix_mean = next(r for r in recs if r["kind"] == "prefix_mean")
                band_start, band_end = spans[mode]
                rank_note = (
                    f"nd={null_dim:<4d} rank={ranks[mode]:<4d}  "
                    f"sv={band_start}..{band_end}"
                )
                print(
                    f"  {label:12s}  {tag}  {rank_note}  "
                    f"prefix mean NLL={prefix_mean['nll']:.3f}  "
                    f"repeat mean NLL={mean_row['nll']:.3f}  acc={mean_row['acc']:.3f}"
                )
                print(
                    "               prefix  "
                    + "  ".join(f"k={r['k']} nll={r['nll']:.3f}" for r in prefix_rows)
                )
                print(
                    "               repeat  "
                    + "  ".join(f"k={r['k']} nll={r['nll']:.3f}" for r in stride_rows)
                )

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")
    if len(null_dims) == 1:
        nd_tag = f"nd{null_dims[0]}"
    else:
        nd_tag = f"nd{null_dims[0]}-{null_dims[-1]}"
    abl_tag = ablation_filename_tag(modes)
    if args.layers is not None:
        stem = f"{model_tag}_{layers_tag(windows[0])}_{nd_tag}_s{args.stride}"
    else:
        stem = (
            f"{model_tag}_L{args.layer_start}-{args.layer_end}_"
            f"sz{args.size}_wst{args.window_stride}_"
            f"{nd_tag}_s{args.stride}"
        )
    out_csv = args.output_dir / f"{stem}_{abl_tag}_swap.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
