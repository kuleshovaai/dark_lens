import json
import pandas as pd
from pathlib import Path
import ast

columns = [
    'id',
    'instruction',
    'question',
    'prompt_template',
    'gold_response',
    'generated_response',
    'response_history',
    'context',
    'prompt',
    'hallucination',
]

def load_dataset(
    file_path: str,
) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file and performs basic preprocessing.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file {file_path}: {e}")
    df = df.dropna(axis=1, how="all")
    missing = [col for col in columns if not col in df.columns]
    if "prompt" in missing:
        if "prompt_template" in missing:
            raise ValueError(
                f"If DataFrame does not contain `prompt` column then "
                f"prompt_template must be provided."
            )
        else:
            def build_prompt(row):
                question = row["question"] if "question" in row.index else None
                context = row["context"] if "context" in row.index else None
                instruction = row["instruction"] if "instruction" in row.index else None
                prompt_template = row["prompt_template"]
                if question is not None:
                    try:
                        question = ast.literal_eval(question)
                    except Exception:
                        pass
                questions_series = None
                if isinstance(question, list):
                    questions_series, question = question, question[0]
                fields = dict()
                fields["question"] = question
                if context is not None:
                    fields["context"] = context
                if instruction is not None:
                    fields["instruction"] = instruction
                prompt = prompt_template.format(**fields)
                if questions_series:
                    return [prompt] + questions_series[1:]
                return prompt
            df["prompt"] = [
                build_prompt(row)
                for _, row in df.iterrows()
            ]
            missing.remove("prompt")
    else:
        try:
            df["prompt"] = df["prompt"].apply(lambda x: ast.literal_eval(x))
        except Exception:
            pass
    if "response_history" not in missing:
        df["response_history"] = df["response_history"].apply(lambda x: ast.literal_eval(x))
    if "generated_response" not in missing:
        df["generated_response"] = df["generated_response"].apply(
            lambda x: ast.literal_eval(x) if x.startswith("[") and x.endswith("]") else x
        )
    if "id" in missing:
        df["id"] = df.index.values
        missing.remove("id")   
    if "hallucination" not in missing:
        df = df[~df["hallucination"].isna()]
    for col in missing:
        df[col] = None
    return df[columns]


def _is_hallucinated_response(item: dict) -> bool:
    labels = item.get("labels") or []
    if labels and isinstance(labels[0], dict):
        return True
    label = item.get("label")
    if label == ["Yes"] or (isinstance(label, list) and "Yes" in label):
        return True
    if labels == ["Yes"] or (isinstance(labels, list) and "Yes" in labels):
        return True
    return False


def load_llama3_8b_spans_dataset(
    spans_path: str,
    source_info_path: str,
    response_model: str = "llama-3-8b-instruct",
    task_type: str | None = None,
) -> pd.DataFrame:
    """Build an annotated-style dataset from Llama 3 8B RAGTruth spans JSONL."""
    source_info_by_id: dict[str, dict] = {}
    with Path(source_info_path).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            source_info_by_id[str(item["source_id"])] = item

    rows = []
    with Path(spans_path).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("model") != response_model:
                continue
            source_id = str(item["source_id"])
            source_info = source_info_by_id.get(source_id)
            if source_info is None:
                continue
            item_task_type = source_info.get("task_type")
            if task_type not in (None, "All") and item_task_type != task_type:
                continue
            rows.append(
                {
                    "id": str(item["id"]),
                    "source_id": source_id,
                    "prompt": source_info["prompt"],
                    "hallucination": int(_is_hallucinated_response(item)),
                    "generated_response": item["response"],
                    "response_history": [],
                    "context": source_info.get("source_info"),
                    "instruction": None,
                    "question": None,
                    "prompt_template": None,
                    "gold_response": None,
                }
            )

    if not rows:
        raise ValueError(
            f"No rows found in {spans_path} for model={response_model!r}"
            + (f" and task_type={task_type!r}" if task_type not in (None, "All") else "")
        )
    return pd.DataFrame(rows)[columns]
