"""ConflictQA answer NLL under within-example dark/middle residual permutation.

Same intervention as ``scripts/nq_swap/swap_dark_matter_nll.py``. Each example
is scored on its own. At the selected positions, the subspace is permuted
among tokens of that same example. Position 0 is left unchanged.
``--positions all`` shuffles every later token; ``--positions prompt``
shuffles prompt tokens after the first; ``--positions response`` shuffles
the answer span. The prompt contains the counter-memory passage. Two fixed
continuations are scored, with no generation:

  * parametric — ``parametric_response`` (model answer with no context)
  * context    — ``counter_memory_response`` (or ``counter_answer`` fallback)

Examples are loaded from ``data/ConflictQA/responses`` CSVs produced by
``generate_parametric_responses.py`` (matched by subset + model name).

Sweeps fixed-size sliding windows over ``[--layer-start, --layer-end]`` with
step ``--stride``. Default: start=0, end=n_layers-2, size=4, stride=size.
``none`` (unhooked) is scored once. The dark subspace is the trailing
``--null-dim`` right-singular directions. The middle subspace is the band in
``mech_interp/wikitext_middle_nll_match`` whose WikiText NLL damage matches
the dark ablation. A window with no middle band is skipped for middle;
dark is still scored. A missing dark band is an error.

Subsets are the configs of ``osunlp/ConflictQA`` (test split only), e.g.
``ConflictQA-popQA-gpt4`` or ``ConflictQA-strategyQA-llama2-7b``.

Example:
  python scripts/conflict_qa/swap_dark_matter_nll.py \\
      --model-name Qwen3-8B --subset ConflictQA-popQA-gpt4 \\
      --null-dim 205 --positions all \\
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

from src.mech_interp.light_matter import (  # noqa: E402
    permute_subspace_and_renormalize,
    unembedding_null_basis,
    unembedding_right_basis,
)
from src.mech_interp.load_model import load_tl_model  # noqa: E402
from src.mech_interp.tokens import encode_prompt_response  # noqa: E402

load_dotenv()
login(token=os.getenv("HUGGING_FACE_API_KEY"))

SWAP_MODES = ("none", "dark", "middle")
ANSWER_KINDS = ("parametric", "context")
NLL_MATCH_DIR = PROJECT_ROOT / "mech_interp" / "wikitext_middle_perm_nll_match"
RESPONSES_DIR = PROJECT_ROOT / "data" / "ConflictQA" / "responses"
SUBSETS = (
    "ConflictQA-popQA-chatgpt",
    "ConflictQA-popQA-gpt4",
    "ConflictQA-popQA-palm2",
    "ConflictQA-popQA-qwen7b",
    "ConflictQA-popQA-llama2-7b",
    "ConflictQA-popQA-llama2-70b",
    "ConflictQA-popQA-vicuna7b",
    "ConflictQA-popQA-vicuna33b",
    "ConflictQA-strategyQA-chatgpt",
    "ConflictQA-strategyQA-gpt4",
    "ConflictQA-strategyQA-palm2",
    "ConflictQA-strategyQA-qwen7b",
    "ConflictQA-strategyQA-llama2-7b",
    "ConflictQA-strategyQA-llama2-70b",
    "ConflictQA-strategyQA-vicuna7b",
    "ConflictQA-strategyQA-vicuna33b",
)

CONTEXT_INSTRUCTION = (
    "Given the context, answer the question as briefly as possible "
    "(a short phrase or name is enough; no extra explanation)."
    "Note that your answer should be strictly based on the given context. "
    'In case the context does not contain the necessary information to answer '
    'the question, please reply with: "Unable to answer based on given context".\n\n'
)
CONTEXT_TEMPLATE = (
    "{instruction}\n\n"
    "Example:\n"
    "Context: The Eiffel Tower is located in Paris, France.\n\n"
    "Question: Where is the Eiffel Tower located?\n\n"
    "Your answer: Paris, France.\n\n"
    "Now answer the following.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Your answer: "
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument(
        "--subset",
        choices=SUBSETS,
        default="ConflictQA-popQA-gpt4",
        help=(
            "ConflictQA subset tag used to resolve "
            "data/ConflictQA/responses/conflict_qa_{subset}_{model}_parametric.csv."
        ),
    )
    p.add_argument(
        "--responses-dir",
        type=Path,
        default=RESPONSES_DIR,
        help="Directory of ConflictQA response CSVs (default: data/ConflictQA/responses).",
    )
    p.add_argument(
        "--responses-csv",
        type=Path,
        default=None,
        help="Explicit response CSV path. Overrides --responses-dir / subset / model.",
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
        help="If set, only this layer set (e.g. 31-34); skips the window sweep.",
    )
    p.add_argument(
        "--null-dim",
        type=int,
        nargs="+",
        default=[256],
        help="Rank(s) of the permuted subspace, e.g. --null-dim 64 128 256",
    )
    p.add_argument(
        "--ablation",
        nargs="+",
        choices=(*SWAP_MODES, "all"),
        default=["all"],
        help=(
            "One or more interventions, e.g. --ablation none dark. "
            "'all' = none once, then dark and the NLL-matched middle per window."
        ),
    )
    p.add_argument(
        "--hook",
        choices=("resid_post", "resid_mid"),
        default="resid_post",
    )
    p.add_argument(
        "--positions",
        choices=("response", "all", "prompt"),
        default="all",
        help=(
            "Permute among answer tokens, prompt tokens after the first token, "
            "or every token after the first. Position 0 is never permuted. "
            "Default: all."
        ),
    )
    p.add_argument(
        "--perm-seed",
        type=int,
        default=0,
        help=(
            "Base seed for in-sample permutations. Combined with example "
            "index and sequence length so layers share one shuffle per forward."
        ),
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=400,
        help="First N ConflictQA rows (default: 400).",
    )
    p.add_argument("--seed", type=int, default=42, help="Unused. Kept for older command lines.")
    p.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "mech_interp/conflict_qa_dark_ablation",
    )
    p.add_argument(
        "--nll-match-dir",
        type=Path,
        default=NLL_MATCH_DIR,
        help=(
            "Directory of match_middle_nll.py CSVs. The middle swap uses the "
            "middle_swap band. A window with no middle_swap row is skipped."
        ),
    )
    p.add_argument(
        "--nll-match-suffix",
        default="se_tol",
        help=(
            "Suffix of the match CSVs after nll_match_ "
            "(default: se_tol, so *_nll_match_se_tol.csv). "
            "Use tol0.01 for *_nll_match_tol0.01.csv."
        ),
    )
    return p.parse_args()


def responses_csv_path(subset: str, model_name: str, responses_dir: Path) -> Path:
    """``ConflictQA-popQA-gpt4`` + ``Qwen3-8B`` → ``conflict_qa_popQA-gpt4_Qwen3-8B_parametric.csv``."""
    if not subset.startswith("ConflictQA-"):
        raise ValueError(f"Unexpected subset name {subset!r}")
    subset_tag = subset.removeprefix("ConflictQA-")
    model_tag = model_name.replace("/", "_")
    return responses_dir / f"conflict_qa_{subset_tag}_{model_tag}_parametric.csv"


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


def parse_layers(spec: str, n_layers: int) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n_layers))
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Empty layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    out = sorted(layers)
    bad = [i for i in out if i < 0 or i >= n_layers]
    if bad:
        raise ValueError(f"Layer indices {bad} outside 0..{n_layers - 1}")
    if not out:
        raise ValueError(f"No layers parsed from {spec!r}")
    return out


def layers_tag(layers: list[int]) -> str:
    if not layers:
        return "Lnone"
    if layers == list(range(layers[0], layers[-1] + 1)):
        return f"L{layers[0]}-{layers[-1]}" if len(layers) > 1 else f"L{layers[0]}"
    if len(layers) > 8:
        return f"L{layers[0]}-{layers[-1]}_n{len(layers)}"
    return "L" + "_".join(str(i) for i in layers)


def apply_chat_template(
    tl_model: HookedTransformer,
    prompt: str,
    *,
    enable_thinking: bool,
) -> str:
    tokenizer = tl_model.tokenizer
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def projector_for_mode(
    mode: str,
    tl_model: HookedTransformer,
    null_dim: int,
) -> torch.Tensor | None:
    device = tl_model.W_U.device
    if mode == "none":
        return None
    if mode == "dark":
        basis = unembedding_null_basis(tl_model.W_U.detach(), null_dim=null_dim)
    else:
        raise ValueError(f"Unknown ablation mode {mode!r}")
    return basis.to(device=device, dtype=torch.float32)


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _context_aligned_response(item: dict[str, Any]) -> str:
    """Prefer model ``counter_memory_response``; fall back to dataset ``counter_answer``."""
    return _text(item.get("counter_memory_response")) or _text(item.get("counter_answer"))


def load_conflict_qa(
    *,
    subset: str,
    model_name: str,
    num_samples: int | None,
    responses_dir: Path,
    responses_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Load ConflictQA rows from a local responses CSV.

    Prompt context is ``counter_memory``. Continuations scored later:

    * parametric — ``parametric_response``
    * context    — ``counter_memory_response`` (else ``counter_answer``)
    """
    path = (
        responses_csv
        if responses_csv is not None
        else responses_csv_path(subset, model_name, responses_dir)
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"ConflictQA responses CSV not found: {path}. "
            "Run scripts/conflict_qa/generate_parametric_responses.py first, "
            "or pass --responses-csv."
        )
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        record = item.to_dict()
        question = _text(record.get("question"))
        context = _text(record.get("counter_memory"))
        parametric = _text(record.get("parametric_response"))
        contextual = _context_aligned_response(record)
        if not (question and context and parametric and contextual):
            continue
        example_id = record.get("example_id", len(rows))
        rows.append(
            {
                "example_id": example_id,
                "question": question,
                "popularity": record.get("popularity"),
                "ground_truth": record.get("ground_truth"),
                "context": context,
                "parametric_answer": parametric,
                "context_answer": contextual,
                "parametric_memory": _text(record.get("parametric_memory")),
                "counter_memory_aligned_evidence": _text(
                    record.get("counter_memory_aligned_evidence")
                ),
                "prompt": CONTEXT_TEMPLATE.format(
                    instruction=CONTEXT_INSTRUCTION,
                    context=context,
                    question=question,
                ),
            }
        )
        if num_samples is not None and len(rows) >= num_samples:
            break
    if num_samples is not None and len(rows) < num_samples:
        raise ValueError(
            f"Requested {num_samples} rows from {path} but only {len(rows)} "
            "have question, counter_memory, parametric_response, and "
            "counter_memory_response/counter_answer."
        )
    return rows


def eligible_positions(
    seq_len: int,
    prompt_len: int,
    *,
    positions: str,
) -> torch.Tensor | None:
    """Indices to shuffle among, or None if fewer than 2.

    Position 0 is never included. ``all`` uses every later token, ``prompt``
    uses prompt tokens after the first, and ``response`` uses the answer span.
    """
    if positions == "prompt":
        idxs = list(range(1, min(prompt_len, seq_len)))
    elif positions == "all":
        idxs = list(range(1, seq_len))
    else:
        if seq_len <= prompt_len:
            return None
        idxs = list(range(prompt_len, seq_len))
    if len(idxs) < 2:
        return None
    return torch.tensor(idxs, dtype=torch.long)


def permute_seed_for(base_seed: int, example_idx: int, seq_len: int) -> int:
    return int((base_seed + example_idx * 1_000_003 + seq_len * 97) % (2**31 - 1))


def make_resid_perm_hook(
    p_subspace: torch.Tensor,
    prompt_len: int,
    *,
    positions: str,
    perm_seed: int,
    example_idx: int,
) -> Callable:
    """Permute the subspace among eligible positions of this example.

    Position 0 is copied through unchanged.
    """

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
    suffix: str = "se_tol",
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


def render_prompt(
    row: dict[str, Any],
    tl_model: HookedTransformer,
    *,
    apply_template: bool,
    enable_thinking: bool,
) -> str:
    if apply_template:
        return apply_chat_template(
            tl_model, row["prompt"], enable_thinking=enable_thinking
        )
    return row["prompt"]


def answer_text(row: dict[str, Any], kind: str) -> str:
    if kind == "parametric":
        return row["parametric_answer"]
    if kind == "context":
        return row["context_answer"]
    raise ValueError(f"Unknown answer kind {kind!r}")


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
        hook_fn = make_resid_perm_hook(
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


def score_examples(
    *,
    tl_model: HookedTransformer,
    examples: list[dict[str, Any]],
    prompts: list[str],
    layers: list[int],
    hook_name: str,
    mode: str,
    null_dim: int | None,
    p_subspace: torch.Tensor | None,
    positions: str,
    perm_seed: int,
    desc: str,
    subspace_rank: int | None = None,
    band_start: int | None = None,
    band_end: int | None = None,
) -> list[dict[str, Any]]:
    """Score every example. The subspace is permuted inside that example."""
    if len(examples) != len(prompts):
        raise ValueError(
            f"examples ({len(examples)}) and prompts ({len(prompts)}) differ"
        )
    label = mode_label(mode)
    records: list[dict[str, Any]] = []
    for example_idx, (row, prompt) in enumerate(
        tqdm(zip(examples, prompts), total=len(examples), desc=desc)
    ):
        for kind in ANSWER_KINDS:
            answer = answer_text(row, kind)
            metrics = answer_nll(
                tl_model,
                prompt,
                answer,
                layers=layers,
                hook_name=hook_name,
                p_subspace=p_subspace,
                positions=positions,
                perm_seed=perm_seed,
                example_idx=example_idx,
            )
            records.append(
                {
                    "example_idx": example_idx,
                    "example_id": row["example_id"],
                    "question": row["question"],
                    "popularity": row["popularity"],
                    "ground_truth": row["ground_truth"],
                    "answer_kind": kind,
                    "answer": answer,
                    "parametric_answer": row["parametric_answer"],
                    "context_answer": row["context_answer"],
                    "ablation": label,
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
                    "perm_seed": "" if mode == "none" else perm_seed,
                    **metrics,
                }
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def ablation_filename_tag(ablation_args: list[str]) -> str:
    if ablation_args == ["all"]:
        return "all"
    return "_".join(ablation_args)


def main() -> None:
    args = parse_args()
    null_dims = list(dict.fromkeys(args.null_dim))
    if any(d <= 0 for d in null_dims):
        raise ValueError(f"--null-dim values must be positive, got {args.null_dim}")
    ablation_args = list(dict.fromkeys(args.ablation))
    if "all" in ablation_args:
        if ablation_args != ["all"]:
            raise ValueError("--ablation 'all' cannot be combined with other modes")
        modes = list(SWAP_MODES)
    else:
        modes = ablation_args
    hooked_modes = [mode for mode in modes if mode != "none"]
    if args.num_samples < 1:
        raise ValueError(f"--num-samples must be positive, got {args.num_samples}")

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
                try:
                    matched_bands[(null_dim, layers[0], layers[-1])] = lookup_nll_match_band(
                        match_table,
                        ablation="middle_swap",
                        null_dim=null_dim,
                        layer_start=layers[0],
                        layer_end=layers[-1],
                    )
                except FileNotFoundError:
                    print(
                        f"  {layers_tag(layers)} nd={null_dim}: no middle_swap "
                        f"in *_nll_match_{args.nll_match_suffix}.csv; "
                        "middle will be skipped for this window"
                    )
        print(
            f"NLL-match bands: {args.nll_match_dir}  "
            f"suffix={args.nll_match_suffix}  "
            f"windows={len(matched_bands)}"
        )

    examples = load_conflict_qa(
        subset=args.subset,
        model_name=args.model_name,
        num_samples=args.num_samples,
        responses_dir=args.responses_dir,
        responses_csv=args.responses_csv,
    )
    print(
        f"Loaded {len(examples)} {args.subset} examples  "
        f"(parametric=parametric_response; context=counter_memory / "
        f"counter_memory_response)  "
        f"windows={len(windows)}  perm_seed={args.perm_seed}"
    )
    print("Layer windows:", ", ".join(layers_tag(w) for w in windows))

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
    subset_tag = args.subset.removeprefix("ConflictQA-")
    abl_tag = ablation_filename_tag(ablation_args)
    if args.layers is not None:
        layer_part = layers_tag(windows[0])
    else:
        layer_part = (
            f"L{args.layer_start}-{args.layer_end}_"
            f"s{args.size}_st{args.stride}"
        )
    name_parts = [f"conflict_qa_{subset_tag}_{model_tag}_perm_nll_{layer_part}"]
    if args.positions != "all":
        name_parts.append(args.positions)
    if "middle" in modes:
        match_tag = (
            "nllmatch"
            if args.nll_match_suffix == "se_tol"
            else f"nllmatch_{args.nll_match_suffix}"
        )
        name_parts.append(match_tag)
    name_parts.append(abl_tag)
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
                p_subspace=None,
                positions=args.positions,
                perm_seed=args.perm_seed,
                desc="none",
            )
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
                    band_key = (null_dim, layers[0], layers[-1])
                    if band_key not in matched_bands:
                        continue
                    start, end, rank = matched_bands[band_key]
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
                            p_subspace=basis_from_vh(match_vh, start, rank, match_device),
                            positions=args.positions,
                            perm_seed=args.perm_seed,
                            desc=f"middle_perm {tag} nd={null_dim} rank={rank}",
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
                        p_subspace=projectors[mode],
                        positions=args.positions,
                        perm_seed=args.perm_seed,
                        desc=f"{mode_label(mode)} {tag} nd={null_dim}",
                    )
                )

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")

    summary = (
        df.groupby(
            ["ablation", "null_dim", "layer_start", "answer_kind"], dropna=False
        )["nll_mean"]
        .mean()
        .unstack("answer_kind")
    )
    print(summary)


if __name__ == "__main__":
    main()
