"""Cities/sports/celebrity yes-no NLL under in-sample dark/middle residual permutation.

No generation. For each question, score both fixed continuations:

  * factual     — ``Yes`` if ``label==1`` else ``No``
  * nonfactual  — the opposite token

Sweeps fixed-size sliding windows over ``[--layer-start, --layer-end]`` with
step ``--stride`` (overlap = size - stride). Default: start=0, end=n_layers-2
(excludes the final block), size=4, stride=size. For each window, permute the
dark or middle subspace among eligible tokens within the sample (skip BOS /
index 0 by default), leave the complement in place, and restore each
position's L2. ``none`` is the unhooked baseline. The dark subspace is the
trailing ``--null-dim`` right-singular directions. The middle subspace is the
band in ``mech_interp/wikitext_middle_nll_match`` (``*_nll_match_tol0.01.csv``
by default) whose WikiText NLL damage matches the dark ablation. A missing
band for this model, hook, null rank, and layer window is an error. Pass
``--layers`` to score a single layer set instead of sweeping.

Examples:
  # Single layer range
  python scripts/factual_qa/swap_dark_matter.py \\
      --dataset cities --layers 24-35 --null-dim 256 --positions all

  # Non-overlapping 4-layer windows over [0, 34]
  python scripts/factual_qa/swap_dark_matter.py \\
      --dataset cities --null-dim 256 --positions all \\
      --layer-start 0 --layer-end 34 --size 4 --stride 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformer_lens import HookedTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablate_dark_matter import (  # noqa: E402
    layers_tag,
    parse_layers,
    projector_for_mode,
    resolve_prompt,
)
from src.mech_interp.light_matter import (  # noqa: E402
    permute_subspace_and_renormalize,
    unembedding_right_basis,
)
from src.mech_interp.load_model import load_tl_model  # noqa: E402
from src.mech_interp.tokens import encode_prompt_response  # noqa: E402
from utils import (  # noqa: E402
    PROJECT_ROOT,
    build_generation_rows,
    load_celebrity_dataset,
    load_cities_dataset,
    load_sports_dataset,
)

load_dotenv()
login(token=os.getenv("HUGGING_FACE_API_KEY"))

SWAP_MODES = ("none", "dark", "middle")
ANSWER_KINDS = ("factual", "nonfactual")
NLL_MATCH_DIR = PROJECT_ROOT / "mech_interp" / "wikitext_middle_perm_nll_match"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="Qwen3-8B")
    p.add_argument("--dataset", choices=("cities", "sports", "celebrity"), default="cities")
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
            "If set, only this layer set (e.g. 24-35 or 'all'); skips the "
            "window sweep."
        ),
    )
    p.add_argument(
        "--null-dim",
        type=int,
        nargs="+",
        default=[256],
        help="Rank(s) of the swapped subspace, e.g. --null-dim 16 32 64",
    )
    p.add_argument(
        "--ablation",
        choices=(*SWAP_MODES, "all"),
        default="all",
        help=(
            "Which intervention(s). 'all' = none once, then dark and the "
            "NLL-matched middle per window."
        ),
    )
    p.add_argument(
        "--hook",
        choices=("resid_post", "resid_mid"),
        default="resid_post",
        help="Residual stream site to patch (block output vs post-attention).",
    )
    p.add_argument(
        "--positions",
        choices=("response", "all"),
        default="all",
        help=(
            "Permute among answer tokens only, or every position except "
            "BOS / first token. Default: all (skip BOS)."
        ),
    )
    p.add_argument(
        "--perm-seed",
        type=int,
        default=0,
        help=(
            "Base seed for in-sample permutations. Combined with example "
            "index and seq length so layers share one shuffle per forward."
        ),
    )
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "mech_interp/",
    )
    p.add_argument(
        "--nll-match-dir",
        type=Path,
        default=NLL_MATCH_DIR,
        help=(
            "Directory of match_middle_nll.py CSVs. The middle permutation "
            "uses the middle_swap band. A missing band aborts the run."
        ),
    )
    p.add_argument(
        "--nll-match-suffix",
        default="tol0.01",
        help=(
            "Suffix of the match CSVs after nll_match_ "
            "(default: tol0.01, so *_nll_match_tol0.01.csv)."
        ),
    )
    return p.parse_args()


def layer_windows(
    layer_start: int,
    layer_end: int,
    size: int,
    stride: int,
) -> list[list[int]]:
    """Fixed-size sliding windows covering ``[layer_start, layer_end]`` inclusive.

    Starts at ``layer_start``, then ``layer_start + stride``, … . If the regular
    grid would leave the end of the interval uncovered, a final window ending
    exactly at ``layer_end`` is appended.
    """
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


def eligible_positions(
    seq_len: int,
    prompt_len: int,
    *,
    positions: str,
) -> torch.Tensor | None:
    """1d long indices to shuffle among, or None if fewer than 2."""
    if positions == "all":
        idxs = list(range(1, seq_len))
    else:  # response
        if seq_len <= prompt_len:
            return None
        idxs = list(range(prompt_len, seq_len))
    if len(idxs) < 2:
        return None
    return torch.tensor(idxs, dtype=torch.long)


def permute_seed_for(base_seed: int, example_idx: int, seq_len: int) -> int:
    return int((base_seed + example_idx * 1_000_003 + seq_len * 97) % (2**31 - 1))


def make_resid_swap_hook(
    p_subspace: torch.Tensor,
    prompt_len: int,
    *,
    positions: str,
    perm_seed: int,
    example_idx: int,
) -> Callable:
    def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
        seq_len = resid.shape[1]
        pos = eligible_positions(seq_len, prompt_len, positions=positions)
        if pos is None:
            return resid
        pos = pos.to(device=resid.device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(permute_seed_for(perm_seed, example_idx, seq_len))
        order = torch.randperm(pos.numel(), generator=gen).to(device=resid.device)
        slice_h = resid.index_select(1, pos)
        swapped = permute_subspace_and_renormalize(
            slice_h, p_subspace, order, dim=1
        )
        patched = resid.clone()
        patched[:, pos, :] = swapped
        return patched

    return hook_fn


def mode_label(mode: str) -> str:
    return "none" if mode == "none" else f"{mode}_perm"


def load_nll_match_table(
    directory: Path,
    model_name: str,
    hook: str,
    suffix: str = "tol0.01",
) -> pd.DataFrame:
    """Middle and dark bands from ``match_middle_nll.py`` for this model and hook."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No NLL-match directory {directory}. "
            "Run scripts/wikitext/match_middle_nll.py first."
        )
    pattern = f"*_nll_match_{suffix}.csv"
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No {pattern} files in {directory}. "
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


def factual_answers(gt_label: Any) -> dict[str, str]:
    """Map answer_kind -> Yes/No string for this example's ground truth."""
    gt = int(gt_label)
    factual = "Yes" if gt == 1 else "No"
    nonfactual = "No" if gt == 1 else "Yes"
    return {"factual": factual, "nonfactual": nonfactual}


@torch.no_grad()
def answer_nll(
    tl_model: HookedTransformer,
    prompt: str,
    answer: str,
    *,
    layers: list[int],
    hook_name: str,
    p_subspace: torch.Tensor | None,
    positions: str,
    perm_seed: int,
    example_idx: int,
) -> dict[str, float | int]:
    tokens, _, response_len, seq_len = encode_prompt_response(
        tl_model, prompt, answer, prediction_aligned=True
    )
    prompt_len = int(tokens.shape[1]) - int(response_len)
    if p_subspace is None:
        logits = tl_model(tokens)
    else:
        hook_fn = make_resid_swap_hook(
            p_subspace,
            prompt_len,
            positions=positions,
            perm_seed=perm_seed,
            example_idx=example_idx,
        )
        fwd_hooks = [
            (f"blocks.{layer}.{hook_name}", hook_fn) for layer in layers
        ]
        with tl_model.hooks(fwd_hooks=fwd_hooks):
            logits = tl_model(tokens)

    logp = F.log_softmax(
        logits[:, prompt_len - 1 : prompt_len + response_len - 1, :].float(),
        dim=-1,
    )
    targets = tokens[:, prompt_len : prompt_len + response_len]
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    hit = logits[:, prompt_len - 1 : prompt_len + response_len - 1, :].argmax(
        dim=-1
    ) == targets
    return {
        "nll_mean": float(nll.mean().item()),
        "nll_sum": float(nll.sum().item()),
        "acc": float(hit.float().mean().item()),
        "response_len": int(response_len),
        "seq_len": int(seq_len),
        "prompt_len": int(prompt_len),
    }


def main() -> None:
    args = parse_args()
    null_dims = list(dict.fromkeys(args.null_dim))
    if any(d <= 0 for d in null_dims):
        raise ValueError(f"--null-dim values must be positive, got {args.null_dim}")
    modes = list(SWAP_MODES) if args.ablation == "all" else [args.ablation]
    hooked_modes = [mode for mode in modes if mode != "none"]

    if args.dataset == "cities":
        df = load_cities_dataset(args.num_samples)
    elif args.dataset == "sports":
        df = load_sports_dataset(args.num_samples)
    elif args.dataset == "celebrity":
        df = load_celebrity_dataset(args.num_samples)
    else:
        raise NotImplementedError
    examples = build_generation_rows(df.to_dict(orient="records"), args.dataset)

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
        match_table = load_nll_match_table(
            args.nll_match_dir,
            args.model_name,
            args.hook,
            suffix=args.nll_match_suffix,
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
            f"suffix={args.nll_match_suffix}  "
            f"windows={len(matched_bands)}"
        )

    print(
        f"Loaded {len(examples)} {args.dataset} questions  "
        f"(Yes/No NLL; in-sample subspace permute; "
        f"windows={len(windows)}; perm_seed={args.perm_seed})"
    )
    print("Layer windows:", ", ".join(layers_tag(w) for w in windows))

    prompts = [
        resolve_prompt(
            row,
            tl_model,
            apply_template=args.apply_chat_template,
            enable_thinking=args.enable_thinking,
        )
        for row in tqdm(examples, desc="render prompts")
    ]

    output_dir = Path(args.output_dir) / f"{args.dataset}_dark_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")
    if args.layers is not None:
        layer_part = layers_tag(windows[0])
    else:
        layer_part = (
            f"L{args.layer_start}-{args.layer_end}_"
            f"s{args.size}_st{args.stride}"
        )
    name_parts = [f"{args.dataset}_{model_tag}_perm_nll_{layer_part}"]
    if "middle" in modes:
        match_tag = (
            "nllmatch"
            if args.nll_match_suffix == "se_tol"
            else f"nllmatch_{args.nll_match_suffix}"
        )
        name_parts.append(match_tag)
    name_parts.append(args.positions)
    name_parts.append(args.ablation)
    out_csv = output_dir / ("_".join(name_parts) + ".csv")

    records: list[dict[str, Any]] = []

    def score_run(
        *,
        mode: str,
        null_dim: int | None,
        p_subspace: torch.Tensor | None,
        layers: list[int],
        desc: str,
        subspace_rank: int | None = None,
        band_start: int | None = None,
        band_end: int | None = None,
    ) -> None:
        label = mode_label(mode)
        for example_idx, (row, prompt) in enumerate(
            tqdm(zip(examples, prompts), total=len(examples), desc=desc)
        ):
            answers = factual_answers(row["gt_response"])
            for kind in ANSWER_KINDS:
                answer = answers[kind]
                metrics = answer_nll(
                    tl_model,
                    prompt,
                    answer,
                    layers=layers,
                    hook_name=hook_name,
                    p_subspace=p_subspace,
                    positions=args.positions,
                    perm_seed=args.perm_seed,
                    example_idx=example_idx,
                )
                records.append(
                    {
                        "example_idx": example_idx,
                        "example_id": row["example_id"],
                        "question": row["question"],
                        "gt_label": int(row["gt_response"]),
                        "answer_kind": kind,
                        "answer": answer,
                        "ablation": label,
                        "layers": (
                            "" if mode == "none" else ",".join(str(i) for i in layers)
                        ),
                        "layer_start": "" if mode == "none" else layers[0],
                        "layer_end": "" if mode == "none" else layers[-1],
                        "hook": args.hook,
                        "positions": "" if mode == "none" else args.positions,
                        "null_dim": 0 if mode == "none" else null_dim,
                        "subspace_rank": "" if subspace_rank is None else subspace_rank,
                        "band_start": "" if band_start is None else band_start,
                        "band_end": "" if band_end is None else band_end,
                        "perm_seed": "" if mode == "none" else args.perm_seed,
                        **metrics,
                    }
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if "none" in modes:
        score_run(
            mode="none",
            null_dim=None,
            p_subspace=None,
            layers=[],
            desc="none",
        )

    for null_dim in null_dims:
        projectors = {
            mode: projector_for_mode(mode, tl_model, null_dim)
            for mode in hooked_modes
            if mode != "middle"
        }
        for layers in windows:
            tag = layers_tag(layers)
            for mode in hooked_modes:
                if mode == "middle":
                    start, end, rank = matched_bands[(null_dim, layers[0], layers[-1])]
                    assert match_vh is not None and match_device is not None
                    score_run(
                        mode="middle",
                        null_dim=null_dim,
                        p_subspace=basis_from_vh(match_vh, start, rank, match_device),
                        layers=layers,
                        desc=f"middle_perm {tag} nd={null_dim} rank={rank}",
                        subspace_rank=rank,
                        band_start=start,
                        band_end=end,
                    )
                    continue
                score_run(
                    mode=mode,
                    null_dim=null_dim,
                    p_subspace=projectors[mode],
                    layers=layers,
                    desc=f"{mode_label(mode)} {tag} nd={null_dim}",
                )

    out = pd.DataFrame(records)
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows to {out_csv}")

    summary = (
        out.groupby(
            ["ablation", "null_dim", "layer_start", "answer_kind"], dropna=False
        )["nll_mean"]
        .mean()
        .unstack("answer_kind")
    )
    if {"factual", "nonfactual"} <= set(summary.columns):
        summary["factual - nonfactual"] = summary["factual"] - summary["nonfactual"]
    print(summary)


if __name__ == "__main__":
    main()
