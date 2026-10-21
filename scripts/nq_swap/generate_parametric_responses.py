"""Elicit parametric (no-context) answers from an LLM on NQ-Swap questions.

Loads full rows from the local NQ-Swap CSV, prompts the model with a few-shot
template that asks for short factual answers and **no context passage**, then
writes a CSV with every original NQ-Swap field plus the generated response.

Example:
  python scripts/nq_swap/generate_parametric_responses.py \\
      --model-name Llama-3.1-8B-Instruct \\
      --num-samples 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import LLM  # noqa: E402

DEFAULT_CSV = PROJECT_ROOT / "data" / "NQSwap" / "nq_swap_old.csv"

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
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV,
        help=f"NQ-Swap CSV (default: {DEFAULT_CSV.relative_to(PROJECT_ROOT)}).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Keep only the first N rows with a non-empty question (default: all).",
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
        default=PROJECT_ROOT / "data" / "NQSwap" / "responses",
    )
    return p.parse_args()


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _csv_cell(value: Any) -> Any:
    """Serialize nested fields so the CSV keeps the full row."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        # original_answers / substituted_answers are often stringified lists
        return value
    return value


def load_nq_swap_questions(
    csv_path: Path,
    num_samples: int | None,
) -> list[dict[str, Any]]:
    """Load full NQ-Swap rows; no context is put in the generation prompt."""
    path = csv_path if csv_path.is_absolute() else PROJECT_ROOT / csv_path
    if not path.exists():
        raise FileNotFoundError(f"NQ-Swap CSV not found: {path}")

    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, item in df.iterrows():
        record = item.to_dict()
        question = _text(record.get("question"))
        if not question:
            continue
        row = {key: _csv_cell(value) for key, value in record.items()}
        row.update(
            {
                "example_id": len(rows),
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
            f"Requested {num_samples} NQ-Swap questions but only {len(rows)} "
            f"have a non-empty question in {path}."
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
    cache_dir = PROJECT_ROOT / "cache" / "nq_swap" / model_name
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
    args = parse_args()

    rows = load_nq_swap_questions(args.csv_path, args.num_samples)
    print(
        f"Loaded {len(rows)} NQ-Swap questions from {args.csv_path}; "
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
    output_path = args.output_dir / f"nq_swap_{model_tag}_parametric.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} parametric responses to {output_path}")


if __name__ == "__main__":
    main()
