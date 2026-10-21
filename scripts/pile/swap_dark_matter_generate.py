"""Generate Pile continuations under position-matched mean dark/middle replace.

Calibration (``--mean-seqs`` random Pile chunks of length ``--chunk-len``):
  at each hooked layer, record the dark/middle residual component at every
  position and average across chunks → ``mean_dm[pos]``.

Generation (disjoint later chunks): take the leading ``--prompt-len`` tokens as
prompt and generate ``--max-new-tokens``. At each step, at hooked layers, replace
the subspace component at absolute position ``k`` with ``mean_dm[k]`` (skip BOS),
keep the complement, renormalize. Prompt positions stay fixed across steps.

Sweeps layer windows via ``--layer-start`` / ``--layer-end`` / ``--size`` /
``--stride`` (or ``--layers``).

Example:
  python scripts/pile/swap_dark_matter_generate.py Qwen3-8B \\
      --layer-start 24 --layer-end 24 --size 1 --stride 1 \\
      --null-dim 256 --ablation none dark \\
      --mean-seqs 50 --prompt-len 64 --max-new-tokens 64 --max-chunks 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformer_lens import HookedTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "nq_swap"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablate_dark_matter_generate import (  # noqa: E402
    layers_tag,
    parse_layers,
    projector_for_mode,
)
from src.mech_interp.load_model import load_tl_model  # noqa: E402
from pile_utils import (  # noqa: E402
    DEFAULT_PILE_DATASET,
    DEFAULT_PILE_SPLIT,
    SWAP_MODES,
    build_pile_chunks,
    dataset_tag,
    layer_windows,
)

load_dotenv()
login(token=os.getenv("HUGGING_FACE_API_KEY"))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mech_interp" / "pile_dark_ablation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_name", help="Model short name, e.g. Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument(
        "--dataset",
        default=DEFAULT_PILE_DATASET,
        help=f"HF datasets name (default: {DEFAULT_PILE_DATASET}).",
    )
    p.add_argument(
        "--dataset-config",
        default=None,
        help="Optional HF config name (Pile mirrors usually need none).",
    )
    p.add_argument(
        "--split",
        default=DEFAULT_PILE_SPLIT,
        help=f"Dataset split (default: {DEFAULT_PILE_SPLIT}).",
    )
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="Load the full split into memory instead of streaming.",
    )
    p.add_argument(
        "--text-field",
        default="text",
        help="Column containing document text (default: text).",
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
        help="If set, only this layer set (e.g. 24); skips the window sweep.",
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
        "--chunk-len",
        type=int,
        default=256,
        help=(
            "Token length for calibration chunks and max absolute position "
            "covered by mean DM (default: 256)."
        ),
    )
    p.add_argument(
        "--mean-seqs",
        type=int,
        default=50,
        help="Calibration chunks used to estimate mean DM per position (default: 50).",
    )
    p.add_argument(
        "--prompt-len",
        type=int,
        default=64,
        help="Leading tokens of each eval chunk used as the generation prompt.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Tokens to generate after the prompt (default: 64).",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=16,
        help="Number of eval chunk prefixes to generate from (default: 16).",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return p.parse_args()


def mode_label(mode: str) -> str:
    return "none" if mode == "none" else f"{mode}_mean"


def subspace_component(hidden: torch.Tensor, p_subspace: torch.Tensor) -> torch.Tensor:
    """Project ``hidden`` onto the subspace spanned by ``p_subspace``."""
    p = p_subspace.to(device=hidden.device, dtype=torch.float32)
    h = hidden.float()
    if p.shape[-1] == p.shape[-2]:
        return h @ p
    return (h @ p) @ p.mT


def replace_subspace_with_mean(
    resid: torch.Tensor,
    p_subspace: torch.Tensor,
    mean_dm: torch.Tensor,
) -> torch.Tensor:
    """Replace subspace component at each position with ``mean_dm[pos]`` (skip BOS).

    ``mean_dm`` is ``[seq, d_model]`` (or longer; truncated to current seq).
    """
    seq_len = resid.shape[1]
    if seq_len <= 1:
        return resid
    if mean_dm.shape[0] < seq_len:
        raise ValueError(
            f"mean_dm covers {mean_dm.shape[0]} positions but seq_len={seq_len}"
        )
    h = resid.float()
    orig_norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(1e-8)
    dark = subspace_component(h, p_subspace)
    mean = mean_dm[:seq_len].to(device=h.device, dtype=h.dtype)
    h_new = h - dark + mean.unsqueeze(0)
    new_norm = torch.linalg.norm(h_new, dim=-1, keepdim=True).clamp_min(1e-8)
    out = (h_new * (orig_norm / new_norm)).to(dtype=resid.dtype)
    patched = resid.clone()
    patched[:, 1:, :] = out[:, 1:, :]
    return patched


@torch.no_grad()
def estimate_mean_dm(
    tl_model,
    chunks: torch.Tensor,
    *,
    layers: list[int],
    hook_name: str,
    p_subspace: torch.Tensor,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Position-matched mean subspace component per layer.

    Returns ``{layer: [seq_len, d_model]}``.
    """
    layers = sorted(set(layers))
    if not layers:
        return {}
    seq_len = int(chunks.shape[1])
    d_model = tl_model.cfg.d_model
    device = tl_model.cfg.device
    sums = {
        layer: torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
        for layer in layers
    }
    n_seen = 0

    def on_resid(resid: torch.Tensor, hook) -> torch.Tensor:
        layer = int(hook.name.split(".")[1])
        if layer not in sums:
            return resid
        if resid.shape[1] != seq_len:
            raise ValueError(
                f"Expected seq_len={seq_len}, got {resid.shape[1]} at {hook.name}"
            )
        dark = subspace_component(resid, p_subspace)
        sums[layer] += dark.sum(dim=0)
        return resid

    fwd_hooks = [
        (f"blocks.{layer}.{hook_name}", on_resid) for layer in layers
    ]
    n = chunks.shape[0]
    for start in tqdm(range(0, n, batch_size), desc="mean dm", leave=False):
        batch = chunks[start : start + batch_size].to(device)
        with tl_model.hooks(fwd_hooks=fwd_hooks):
            _ = tl_model(batch)
        n_seen += batch.shape[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if n_seen <= 0:
        raise RuntimeError("No calibration chunks for mean DM")
    return {layer: total / n_seen for layer, total in sums.items()}


def make_mean_dm_hooks(
    layers: list[int],
    hook_name: str,
    p_subspace: torch.Tensor,
    mean_dm: dict[int, torch.Tensor],
) -> list[tuple[str, Callable]]:
    by_layer = {layer: mean_dm[layer] for layer in layers}

    def make_hook(layer: int):
        mean = by_layer[layer]

        def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
            return replace_subspace_with_mean(resid, p_subspace, mean)

        return hook_fn

    return [
        (f"blocks.{layer}.{hook_name}", make_hook(layer)) for layer in sorted(by_layer)
    ]


@torch.no_grad()
def generate_one(
    tl_model: HookedTransformer,
    prompt_tokens: torch.Tensor,
    *,
    fwd_hooks: list[tuple[str, Callable]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    """``prompt_tokens`` shape ``[1, prompt_len]``."""
    prompt_len = int(prompt_tokens.shape[1])
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else 0.0,
        "stop_at_eos": True,
        "verbose": False,
        "prepend_bos": False,
        # Full re-forward so absolute positions stay aligned with mean_dm[k].
        "use_past_kv_cache": False,
    }
    if fwd_hooks:
        with tl_model.hooks(fwd_hooks=fwd_hooks):
            out = tl_model.generate(prompt_tokens, **gen_kwargs)
    else:
        out = tl_model.generate(prompt_tokens, **gen_kwargs)
    response_ids = out[0, prompt_len:]
    return tl_model.to_string(response_ids)


def main() -> None:
    args = parse_args()
    null_dims = list(dict.fromkeys(args.null_dim))
    if any(d <= 0 for d in null_dims):
        raise ValueError(f"--null-dim values must be positive, got {args.null_dim}")
    if args.prompt_len < 2:
        raise ValueError(f"--prompt-len must be >= 2, got {args.prompt_len}")
    if args.mean_seqs < 1:
        raise ValueError(f"--mean-seqs must be >= 1, got {args.mean_seqs}")
    if args.max_new_tokens < 1:
        raise ValueError(f"--max-new-tokens must be >= 1, got {args.max_new_tokens}")
    if args.max_chunks < 1:
        raise ValueError(f"--max-chunks must be >= 1, got {args.max_chunks}")
    need_len = args.prompt_len + args.max_new_tokens
    if args.chunk_len < need_len:
        raise ValueError(
            f"--chunk-len={args.chunk_len} must be >= prompt_len+max_new_tokens="
            f"{need_len} so mean_dm covers every generated position"
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
    tokenizer = tl_model.tokenizer
    if tokenizer is None:
        raise RuntimeError("Model has no tokenizer; cannot pack Pile chunks")

    # Calibration chunks first, then disjoint eval chunks for generation.
    packed = build_pile_chunks(
        tokenizer,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        seq_len=args.chunk_len,
        max_chunks=args.mean_seqs + args.max_chunks,
        streaming=not args.no_streaming,
        text_field=args.text_field,
    )
    if packed.shape[0] < args.mean_seqs + args.max_chunks:
        raise ValueError(
            f"Need {args.mean_seqs + args.max_chunks} chunks, got {packed.shape[0]}"
        )
    cal_chunks = packed[: args.mean_seqs]
    eval_chunks = packed[args.mean_seqs : args.mean_seqs + args.max_chunks]
    prompts = eval_chunks[:, : args.prompt_len]

    union_layers = sorted({layer for window in windows for layer in window})
    ds_tag = dataset_tag(args.dataset, args.dataset_config, args.split)
    print(
        f"{args.model_name}  {ds_tag}  "
        f"mean_seqs={cal_chunks.shape[0]}  eval_chunks={prompts.shape[0]}  "
        f"chunk_len={args.chunk_len}  prompt_len={args.prompt_len}  "
        f"max_new_tokens={args.max_new_tokens}  windows={len(windows)}  "
        f"null_dims={null_dims}"
    )
    print("Layer windows:", ", ".join(layers_tag(w) for w in windows))

    rows: list[dict[str, Any]] = []
    prompt_texts = [
        tl_model.to_string(prompts[i]) for i in range(prompts.shape[0])
    ]

    def run_condition(
        *,
        ablation: str,
        null_dim: int,
        layers: list[int],
        fwd_hooks: list[tuple[str, Callable]],
    ) -> None:
        label = mode_label(ablation) if ablation != "none" else "none"
        tag = "" if not layers else layers_tag(layers)
        layer_start = "" if not layers else layers[0]
        layer_end = "" if not layers else layers[-1]
        for i in tqdm(range(prompts.shape[0]), desc=f"{label} {tag or '—'}"):
            prompt_tok = prompts[i : i + 1].to(tl_model.cfg.device)
            continuation = generate_one(
                tl_model,
                prompt_tok,
                fwd_hooks=fwd_hooks,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
            )
            rows.append(
                {
                    "chunk_idx": i,
                    "ablation": label,
                    "null_dim": null_dim,
                    "layers": tag,
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                    "hook": args.hook,
                    "method": "mean_dm" if layers else "",
                    "mean_seqs": args.mean_seqs if layers else 0,
                    "prompt_len": args.prompt_len,
                    "max_new_tokens": args.max_new_tokens,
                    "chunk_len": args.chunk_len,
                    "do_sample": int(args.do_sample),
                    "temperature": args.temperature if args.do_sample else 0.0,
                    "prompt": prompt_texts[i],
                    "continuation": continuation,
                    "dataset": args.dataset,
                    "dataset_config": args.dataset_config or "",
                    "split": args.split,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if "none" in modes:
        run_condition(ablation="none", null_dim=0, layers=[], fwd_hooks=[])

    for null_dim in null_dims:
        projectors = {
            mode: projector_for_mode(mode, tl_model, null_dim) for mode in hooked_modes
        }
        # Estimate means once per mode over the union of window layers.
        mean_by_mode = {
            mode: estimate_mean_dm(
                tl_model,
                cal_chunks,
                layers=union_layers,
                hook_name=hook_name,
                p_subspace=projectors[mode],
                batch_size=args.batch_size,
            )
            for mode in hooked_modes
        }
        for layers in windows:
            for mode in hooked_modes:
                fwd_hooks = make_mean_dm_hooks(
                    layers, hook_name, projectors[mode], mean_by_mode[mode]
                )
                run_condition(
                    ablation=mode,
                    null_dim=null_dim,
                    layers=layers,
                    fwd_hooks=fwd_hooks,
                )

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")
    if len(null_dims) == 1:
        nd_tag = f"nd{null_dims[0]}"
    else:
        nd_tag = f"nd{null_dims[0]}-{null_dims[-1]}"
    if args.layers is not None:
        out_csv = (
            args.output_dir
            / (
                f"{model_tag}_{ds_tag}_"
                f"{layers_tag(windows[0])}_{nd_tag}_"
                f"p{args.prompt_len}_g{args.max_new_tokens}_"
                f"n{prompts.shape[0]}_ms{args.mean_seqs}_mean_gen.csv"
            )
        )
    else:
        out_csv = (
            args.output_dir
            / (
                f"{model_tag}_{ds_tag}_"
                f"L{args.layer_start}-{args.layer_end}_"
                f"sz{args.size}_st{args.stride}_{nd_tag}_"
                f"p{args.prompt_len}_g{args.max_new_tokens}_"
                f"n{prompts.shape[0]}_ms{args.mean_seqs}_mean_gen.csv"
            )
        )
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
