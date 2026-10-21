"""Elicit parametric (no-context) answers from an LLM on ConflictQA questions.

Loads full rows from ``osunlp/ConflictQA``, prompts the model with a few-shot
template that asks for short factual answers and **no context passage**, then
writes a CSV with every original ConflictQA field plus the generated response.

Example:
  python scripts/conflict_qa/generate_parametric_responses.py \\
      --model-name Llama-3.1-8B-Instruct \\
      --subset ConflictQA-popQA-gpt4 \\
      --num-samples 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import LLM  # noqa: E402

HF_DATASET = "osunlp/ConflictQA"
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

PARAMETRIC_INSTRUCTION = (
    "Answer each question with a short factual answer "
    "(a name, place, occupation, or short phrase). "
    "Do not explain. Do not use complete sentences unless necessary."
)

# Few-shot exemplars: short closed-form answers, no retrieved context.
PARAMETRIC_TEMPLATE = (
    "{instruction}\n\n"
    "Example:\n"
    "Question: Where is the Eiffel Tower located?\n"
    "Your answer: Paris, France\n\n"
    "Example:\n"
    "Question: Who wrote Hamlet?\n"
    "Your answer: William Shakespeare\n\n"
    "Example:\n"
    "Question: What is the capital of Japan?\n"
    "Your answer: Tokyo\n\n"
    "Now answer the following.\n\n"
    "Question: {question}\n"
    "Your answer: "
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-name",
        default="Llama-3.1-8B-Instruct",
        help="Short HF model name used by src.llm.",
    )
    p.add_argument(
        "--subset",
        choices=SUBSETS,
        default="ConflictQA-popQA-gpt4",
        help="osunlp/ConflictQA config to take questions from.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Keep only the first N eligible questions (default: all).",
    )
    p.add_argument(
        "--single-gt-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only examples with a single ground-truth answer (default: true).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0 for deterministic parametric answers).",
    )
    p.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ConflictQA" / "responses",
    )
    return p.parse_args()


def subset_filename(subset: str) -> str:
    """``ConflictQA-popQA-gpt4`` → ``conflictQA-popQA-gpt4.json``."""
    if not subset.startswith("ConflictQA-"):
        raise ValueError(f"Unexpected subset name {subset!r}")
    return "conflictQA-" + subset[len("ConflictQA-") :] + ".json"


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _ground_truth_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [_text(v) for v in value if _text(v)]
    text = _text(value)
    return [text] if text else []


def _csv_cell(value: Any) -> Any:
    """Serialize nested ConflictQA fields so the CSV keeps the full row."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def load_conflict_qa_questions(
    subset: str,
    num_samples: int | None,
    *,
    single_gt_only: bool,
) -> list[dict[str, Any]]:
    """Load full ConflictQA rows; no context is put in the generation prompt."""
    path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=subset_filename(subset),
        repo_type="dataset",
    )
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = _text(item.get("question"))
            if not question:
                continue
            gt_list = _ground_truth_list(item.get("ground_truth"))
            if single_gt_only and len(gt_list) != 1:
                continue
            row = {key: _csv_cell(value) for key, value in item.items()}
            row.update(
                {
                    "example_id": len(rows),
                    "subset": subset,
                    "ground_truth_n": len(gt_list),
                    "prompt": PARAMETRIC_TEMPLATE.format(
                        instruction=PARAMETRIC_INSTRUCTION,
                        question=question,
                    ),
                }
            )
            rows.append(row)
            if num_samples is not None and len(rows) >= num_samples:
                break
    if num_samples is not None and len(rows) < num_samples:
        raise ValueError(
            f"Requested {num_samples} {subset} questions but only {len(rows)} "
            f"eligible (single_gt_only={single_gt_only})."
        )
    return rows


def generate_responses(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    apply_chat_template: bool,
    enable_thinking: bool,
) -> None:
    cache_dir = PROJECT_ROOT / "cache" / "conflict_qa" / model_name
    llm = LLM(model_name=model_name, device=device, cache_dir=str(cache_dir))

    prompts = [row["prompt"] for row in rows]
    stream = llm.generate_llm_responses(
        user_prompt=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        apply_chat_template=apply_chat_template,
        enable_thinking=enable_thinking,
        num_return_sequences=1,
    )

    for row, query in tqdm(
        zip(rows, stream),
        total=len(rows),
        desc="Generating parametric responses",
    ):
        row["rendered_prompt"] = query["prompt"]
        row["parametric_response"] = query["response"][0]
        row["model_name"] = model_name


def main() -> None:
    load_dotenv()
    login(token=os.getenv("HUGGING_FACE_API_KEY"))
    args = parse_args()

    rows = load_conflict_qa_questions(
        args.subset,
        args.num_samples,
        single_gt_only=args.single_gt_only,
    )
    print(
        f"Loaded {len(rows)} {args.subset} questions "
        f"(single_gt_only={args.single_gt_only}); "
        f"eliciting parametric answers from {args.model_name}"
    )

    generate_responses(
        rows,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        apply_chat_template=args.apply_chat_template,
        enable_thinking=args.enable_thinking,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")
    subset_tag = args.subset.replace("ConflictQA-", "")
    output_path = (
        args.output_dir
        / f"conflict_qa_{subset_tag}_{model_tag}_parametric.csv"
    )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} parametric responses to {output_path}")


if __name__ == "__main__":
    main()
