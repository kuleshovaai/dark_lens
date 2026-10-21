import re
import pandas as pd
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CITIES_PROMPT = "You are a helpful geography expert. Answer ''Yes'' or ''No''.\nHere is the question: {question}"
SPORTS_PROMPT = "You are a helpful sports expert who can help me. Answer ''Yes'' or ''No''.\nHere is the question: {question}"
CELEBRITY_PROMPT = "You are a helpful celebrity expert who can help me. Answer ''Yes'' or ''No''.\nHere is the question: {question}"

PROMPTS = {"cities": CITIES_PROMPT, "sports": SPORTS_PROMPT, "celebrity": CELEBRITY_PROMPT}

def extract_last_answer(text):
    try:
        
        # matches = re.findall(r'\d+(?:,\d+)*', text)  
        matches = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
        matches = [float(match.replace(",", "")) for match in matches]

        # matches = [int(match.replace(",", "")) for match in matches]

        last_answer = matches[-1] if matches else None
    except ValueError as e:
        print(f"Error: {e}")
        last_answer = None
    return last_answer

def convert_to_question(sentence):
    # Lowercase the first character if it's uppercase
    if sentence[0].isupper():
        sentence = sentence[0].lower() + sentence[1:]
    
    # Locate 'is' and build the question
    parts = sentence.split()
    is_index = parts.index('is')  # Find the index of 'is'
    
    if is_index != -1:
        # Move 'is' to the beginning and rearrange the parts
        parts.pop(is_index)
        question = 'Is ' + ' '.join(parts)
        
        # Replace the final period with a question mark
        if question.endswith('.'):
            question = question[:-1] + '?'
        else:
            question += '?'
        return question
    
    return sentence  # Return the original if 'is' is not found


def load_cities_dataset(num_samples: int, input_csv: str | None = None):
    if input_csv is not None:
        print(f"Loading Cities data from {input_csv}")
        path = Path(input_csv)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        df = pd.read_csv(path)
    else:
        path = PROJECT_ROOT / "data/Cities/cities.csv"
        print(f"Loading Cities data from the default path: {path}")
        df = pd.read_csv(path)

    num_samples = min(num_samples, len(df))
    df = df[:num_samples]
    df["question"] = df.apply(lambda x: convert_to_question(x["statement"]), axis=1)
    return df


def load_sports_dataset(num_samples: int, input_csv: str | None = None):
    if input_csv is not None:
        print(f"Loading Sports data from {input_csv}")
        path = Path(input_csv)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        df = pd.read_csv(path)
    else:
        path = PROJECT_ROOT / "data/Sports/sports_qa.csv"
        print(f"Loading Sports data from the default path: {path}")
        df = pd.read_csv(path)

    num_samples = min(num_samples, len(df))
    df = df[:num_samples]
    return df


def load_celebrity_dataset(num_samples: int, input_csv: str | None = None):
    """Same columns as sports, except ``label`` is derived from ``answer``."""
    if input_csv is not None:
        print(f"Loading Celebrity data from {input_csv}")
        path = Path(input_csv)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        df = pd.read_csv(path)
    else:
        path = PROJECT_ROOT / "data/Celebrities/celebrity_qa.csv"
        print(f"Loading Celebrity data from the default path: {path}")
        df = pd.read_csv(path)

    if "label" not in df.columns:
        answer = df["answer"]
        if answer.dtype == bool:
            labels = answer.astype(int)
        else:
            labels = (
                answer.astype(str).str.strip().str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
            )
        if labels.isna().any():
            bad = answer[labels.isna()].unique().tolist()
            raise ValueError(f"Could not map celebrity answers to labels: {bad}")
        df = df.copy()
        df["label"] = labels.astype(int)

    num_samples = min(num_samples, len(df))
    df = df[:num_samples]
    return df


def build_generation_rows(samples: list[dict[str, Any]], ds_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompt_template = PROMPTS[ds_name]
    for sample in samples:
        prompt = prompt_template.format(
            question=sample["question"],
        )
        row = {
            "example_id": sample["example_id"],
            "question": sample["question"],
            "prompt_template" : prompt_template,
            "prompt": prompt,
            "gt_response": sample['label']
        }
        rows.append(row)
    return rows


