"""NQ-Swap answer NLL under dark/middle subspace shrinkage.

Same scoring setup as ``enhance_dark_matter_nll.py``, with the opposite
reweight. At the selected blocks, split each residual as ``h = h_sub + h_comp``
and replace it by ``ĥ = (1 - α) h_sub + (1 + α) h_comp``, then restore the
original L2 norm. ``α > 0`` shrinks the patched subspace and enlarges its
complement. No generation. Two fixed continuations of the same fake-context
prompt are scored, as in ``swap_dark_matter_nll.py``:

  * parametric — ``parametric_response``
  * context    — ``substituted_answer``
  * distractor — ``distractor`` (only when non-empty)

Examples are loaded from ``data/NQSwap/responses`` CSVs (matched by model
name), same as ``swap_dark_matter_nll.py``.

``none`` (unhooked) is scored once. The dark subspace is the trailing
``--null-dim`` right-singular directions. The middle subspace is the band in
``mech_interp/wikitext_middle_nll_match`` (``*_nll_match_tol0.01.csv`` by
default) whose WikiText NLL matches the dark ablation. A missing band for
this model, hook, null rank, and layer window is an error.
``--positions response`` patches the answer tokens only.

Example:
  python scripts/nq_swap/shrink_dark_matter_nll.py \\
      --model-name Qwen3-8B --layers 20-23 --null-dim 205 --alpha 0.2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

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

from ablate_dark_matter_generate import (  # noqa: E402
    layers_tag,
    parse_layers,
    projector_for_mode,
)
from enhance_dark_matter_generate import make_resid_hook  # noqa: E402
from src.mech_interp.light_matter import unembedding_right_basis  # noqa: E402
from src.mech_interp.load_model import load_tl_model  # noqa: E402
from src.mech_interp.tokens import encode_prompt_response  # noqa: E402
from swap_dark_matter_nll import (  # noqa: E402
    NLL_MATCH_DIR,
    RESPONSES_DIR,
    answer_kinds_for,
    answer_text,
    basis_from_vh,
    load_nll_match_table,
    load_nq_swap_parametric_examples,
    lookup_nll_match_band,
    render_prompt,
    responses_csv_path,
)
from utils import PROJECT_ROOT  # noqa: E402

load_dotenv()
login(token=os.getenv("HUGGING_FACE_API_KEY"))

SHRINK_MODES = ("none", "dark", "middle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument(
        "--layers",
        required=True,
        help=(
            "Transformer blocks to patch. Comma-separated indices and/or "
            "inclusive ranges, or 'all'. Examples: 20-28  |  3,7,20-22  |  all"
        ),
    )
    p.add_argument(
        "--null-dim",
        type=int,
        nargs="+",
        default=[256],
        help="Rank(s) of the shrunk subspace, e.g. --null-dim 256",
    )
    p.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.2],
        help=(
            "Reweight strength(s). ĥ = (1-α) h_sub + (1+α) h_comp "
            "(default: 0.2). α > 0 shrinks the subspace and enlarges "
            "its complement."
        ),
    )
    p.add_argument(
        "--ablation",
        nargs="+",
        choices=(*SHRINK_MODES, "all"),
        default=["all"],
        help="One or more interventions. 'all' = none once, then dark and middle.",
    )
    p.add_argument(
        "--hook",
        choices=("resid_post", "resid_mid"),
        default="resid_post",
        help="Residual stream site to patch (block output vs post-attention).",
    )
    p.add_argument(
        "--positions",
        choices=("response", "all", "prompt"),
        default="all",
        help=(
            "Which residual positions to reweight: answer tokens only; "
            "every token except BOS; or prompt tokens except BOS."
        ),
    )
    p.add_argument("--split", default="dev")
    p.add_argument(
        "--num-samples",
        type=int,
        default=400,
        help=(
            "First N eligible rows from the model parametric responses CSV "
            "(default: 400)."
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="NQ-Swap sample order seed.")
    p.add_argument(
        "--responses-dir",
        type=Path,
        default=RESPONSES_DIR,
        help="Directory of NQ-Swap response CSVs (default: data/NQSwap/responses).",
    )
    p.add_argument(
        "--responses-csv",
        type=Path,
        default=None,
        help="Explicit response CSV path. Overrides --responses-dir / model.",
    )
    p.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument(
        "--nll-match-dir",
        type=Path,
        default=NLL_MATCH_DIR,
        help=(
            "Directory of match_middle_nll.py CSVs. The middle ablation uses "
            "the middle_swap band."
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
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "mech_interp/nq_swap_dark_shrink",
    )
    return p.parse_args()


def ablation_filename_tag(ablation_args: list[str]) -> str:
    if ablation_args == ["all"]:
        return "all"
    return "_".join(ablation_args)


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
    alpha: float,
) -> dict[str, float | int]:
    tokens, _, response_len, seq_len = encode_prompt_response(
        tl_model, prompt, answer, prediction_aligned=True
    )
    prompt_len = int(tokens.shape[1]) - int(response_len)
    if p_subspace is None:
        logits = tl_model(tokens)
    else:
        # Positive α here is the shrink strength. The shared hook implements
        # (1+α) on the subspace, so the opposite sign shrinks it.
        hook_fn = make_resid_hook(
            p_subspace, prompt_len, positions=positions, alpha=-alpha
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


def score_examples(
    *,
    tl_model: HookedTransformer,
    examples: list[dict[str, Any]],
    prompts: list[str],
    layers: list[int],
    hook_name: str,
    mode: str,
    null_dim: int | None,
    alpha: float | None,
    p_subspace: torch.Tensor | None,
    positions: str,
    desc: str,
    subspace_rank: int | None = None,
    band_start: int | None = None,
    band_end: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example_idx, (row, prompt) in enumerate(
        tqdm(zip(examples, prompts), total=len(examples), desc=desc)
    ):
        for kind in answer_kinds_for(row):
            answer = answer_text(row, kind)
            metrics = answer_nll(
                tl_model,
                prompt,
                answer,
                layers=layers,
                hook_name=hook_name,
                p_subspace=p_subspace,
                positions=positions,
                alpha=0.0 if alpha is None else alpha,
            )
            records.append(
                {
                    "example_idx": example_idx,
                    "example_id": row.get("example_id", row.get("id")),
                    "question": row.get("question"),
                    "substitution_type": row.get("substitution_type"),
                    "answer_kind": kind,
                    "answer": answer,
                    "original_answer": row.get("original_answer"),
                    "parametric_response": row.get("parametric_response"),
                    "substituted_answer": row.get("substituted_answer"),
                    "distractor": row.get("distractor"),
                    "ablation": mode,
                    "layers": (
                        "" if mode == "none" else ",".join(str(i) for i in layers)
                    ),
                    "layer_start": "" if mode == "none" else layers[0],
                    "layer_end": "" if mode == "none" else layers[-1],
                    "hook": hook_name.removeprefix("hook_"),
                    "positions": positions,
                    "null_dim": 0 if mode == "none" else null_dim,
                    "subspace_rank": "" if subspace_rank is None else subspace_rank,
                    "band_start": "" if band_start is None else band_start,
                    "band_end": "" if band_end is None else band_end,
                    "alpha": "" if mode == "none" else alpha,
                    **metrics,
                }
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def main() -> None:
    args = parse_args()
    null_dims = list(dict.fromkeys(args.null_dim))
    alphas = list(dict.fromkeys(args.alpha))
    if any(d <= 0 for d in null_dims):
        raise ValueError(f"--null-dim values must be positive, got {args.null_dim}")
    if any(alpha < 0 for alpha in alphas):
        raise ValueError(
            f"--alpha must be >= 0; positive values shrink the subspace, got {args.alpha}"
        )
    ablation_args = list(dict.fromkeys(args.ablation))
    if "all" in ablation_args:
        if ablation_args != ["all"]:
            raise ValueError("--ablation 'all' cannot be combined with other modes")
        modes = list(SHRINK_MODES)
    else:
        modes = ablation_args
    hooked_modes = [mode for mode in modes if mode != "none"]

    tl_model = load_tl_model(args.model_name, args.device, args.attn_impl)
    layers = parse_layers(args.layers, tl_model.cfg.n_layers)
    hook_name = f"hook_{args.hook}"

    matched_bands: dict[int, tuple[int, int, int]] = {}
    match_vh = None
    match_device = None
    if "middle" in modes:
        if layers != list(range(layers[0], layers[-1] + 1)):
            raise ValueError(
                f"--layers {args.layers!r} is not a contiguous window; "
                "NLL-match bands are keyed by layer_start and layer_end."
            )
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
            matched_bands[null_dim] = lookup_nll_match_band(
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

    examples = load_nq_swap_parametric_examples(
        model_name=args.model_name,
        num_samples=args.num_samples,
        responses_dir=args.responses_dir,
        responses_csv=args.responses_csv,
        split=args.split,
        seed=args.seed,
    )
    source_csv = (
        args.responses_csv
        if args.responses_csv is not None
        else responses_csv_path(args.model_name, args.responses_dir)
    )
    n_with_distractor = sum(1 for row in examples if answer_text(row, "distractor"))
    print(
        f"Loaded {len(examples)} NQ-Swap examples from {source_csv.name}  "
        f"(parametric=parametric_response; context=substituted_answer; "
        f"distractor non-empty: {n_with_distractor})  "
        f"layers={layers_tag(layers)}"
    )

    prompts = [
        render_prompt(
            row,
            tl_model,
            apply_template=args.apply_chat_template,
            enable_thinking=args.enable_thinking,
        )
        for row in tqdm(examples, desc="render prompts")
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")
    alpha_tag = "a" + "_".join(f"{alpha:g}" for alpha in alphas)
    name_parts = [
        f"nq_swap_{model_tag}_shrink_nll_{layers_tag(layers)}_{args.positions}_{alpha_tag}"
    ]
    if "middle" in modes:
        match_tag = (
            "nllmatch"
            if args.nll_match_suffix == "se_tol"
            else f"nllmatch_{args.nll_match_suffix}"
        )
        name_parts.append(match_tag)
    name_parts.append(ablation_filename_tag(ablation_args))
    out_csv = args.output_dir / ("_".join(name_parts) + ".csv")

    records: list[dict[str, Any]] = []
    if "none" in modes:
        records.extend(
            score_examples(
                tl_model=tl_model,
                examples=examples,
                prompts=prompts,
                layers=[],
                hook_name=hook_name,
                mode="none",
                null_dim=None,
                alpha=None,
                p_subspace=None,
                positions=args.positions,
                desc="none",
            )
        )

    for alpha in alphas:
        for null_dim in null_dims:
            projectors = {
                mode: projector_for_mode(mode, tl_model, null_dim)
                for mode in hooked_modes
                if mode != "middle"
            }
            for mode in hooked_modes:
                if mode == "middle":
                    start, end, rank = matched_bands[null_dim]
                    assert match_vh is not None and match_device is not None
                    records.extend(
                        score_examples(
                            tl_model=tl_model,
                            examples=examples,
                            prompts=prompts,
                            layers=layers,
                            hook_name=hook_name,
                            mode="middle",
                            null_dim=null_dim,
                            alpha=alpha,
                            p_subspace=basis_from_vh(
                                match_vh, start, rank, match_device
                            ),
                            positions=args.positions,
                            desc=(
                                f"middle nd={null_dim} rank={rank} "
                                f"sv={start}..{end} a={alpha:g}"
                            ),
                            subspace_rank=rank,
                            band_start=start,
                            band_end=end,
                        )
                    )
                    continue
                records.extend(
                    score_examples(
                        tl_model=tl_model,
                        examples=examples,
                        prompts=prompts,
                        layers=layers,
                        hook_name=hook_name,
                        mode=mode,
                        null_dim=null_dim,
                        alpha=alpha,
                        p_subspace=projectors[mode],
                        positions=args.positions,
                        desc=f"{mode} nd={null_dim} a={alpha:g}",
                    )
                )

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")

    summary = (
        df.groupby(
            ["ablation", "null_dim", "alpha", "answer_kind"], dropna=False
        )["nll_mean"]
        .mean()
        .unstack("answer_kind")
    )
    print(summary)


if __name__ == "__main__":
    main()
