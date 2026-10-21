"""Shared helpers for NQ-Swap compute / plot scripts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONDITIONS = ("no_context", "fake_context")

MODELS = [
    {"key": "Llama-3.1-8B-Instruct", "label": "Llama-3.1-8B", "color": "#F4A261"},
    {"key": "Qwen3-8B", "label": "Qwen3-8B", "color": "#9B5DE5"},
]

COND_COLORS = {
    "no_context": "#E76F51",
    "fake_context": "#2A9D8F",
}

METRIC_COLORS = {
    "light_matter": "#2A9D8F",
    "jsd": "#E76F51",
    "l2_norm": "#1D4ED8",
}

# Expected higher score for condition separation / aligned deltas.
METRIC_SPECS = {
    "light_matter": {
        "label": "Light matter",
        "short": "LM",
        "value_col": "light_matter_ratio",
        "positive_condition": "no_context",
        "delta_sign": -1,  # no - fake
        "delta_label": r"LM $\Delta$ (no$-$fake)",
        "file_suffix": "_by_layer.csv",
    },
    "jsd": {
        "label": "FFN logit-lens JSD",
        "short": "JSD",
        "value_col": "logit_lens_jsd",
        "positive_condition": "no_context",
        "delta_sign": -1,  # no - fake
        "delta_label": r"JSD $\Delta$ (no$-$fake)",
        "file_suffix": "_jsd_by_layer.csv",
    },
    "l2_norm": {
        "label": "Residual L2",
        "short": "L2",
        "value_col": "l2_norm",
        "positive_condition": "no_context",
        "delta_sign": -1,  # no - fake
        "delta_label": r"L2 $\Delta$ (no$-$fake)",
        "file_suffix": "_by_layer.csv",
    },
}

LM_FILE_SUFFIX = {
    "blocks": "_by_layer.csv",
    "layers": "_detailed_by_layer.csv",
}


def metric_specs_for_granularity(granularity: str) -> dict:
    """Return METRIC_SPECS with LM suffix set; ``layers`` is LM-only (att/mlp)."""
    if granularity not in LM_FILE_SUFFIX:
        raise ValueError(f"Unknown granularity {granularity!r}; expected blocks or layers")
    specs = {key: dict(spec) for key, spec in METRIC_SPECS.items()}
    specs["light_matter"]["file_suffix"] = LM_FILE_SUFFIX[granularity]
    if granularity == "layers":
        return {"light_matter": specs["light_matter"]}
    return specs


def metric_csv_path(data_dir: Path, model_key: str, metric_key: str, granularity: str = "blocks") -> Path:
    specs = metric_specs_for_granularity(granularity)
    if metric_key not in specs:
        raise KeyError(f"Metric {metric_key!r} not available for granularity={granularity}")
    return data_dir / f"nq_swap_{model_key}{specs[metric_key]['file_suffix']}"


def apply_layer_xticks(ax, n_points: int, granularity: str) -> None:
    """Label x-axis as block indices, or ``{i}.att`` / ``{i}.mlp`` for detailed LM."""
    if granularity == "blocks":
        ax.set_xlabel("Layer")
        return
    labels = [f"{i // 2}.{'att' if i % 2 == 0 else 'mlp'}" for i in range(n_points)]
    ax.set_xticks(range(n_points))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel("Layer")


def load_pairs(input_csv: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(input_csv)
    required = {
        "example_id",
        "condition",
        "rendered_prompt",
        "generated_response",
        "question",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")

    grouped: dict[str, dict[str, list[pd.Series]]] = defaultdict(
        lambda: {c: [] for c in CONDITIONS}
    )
    for _, row in df.iterrows():
        condition = str(row["condition"])
        if condition not in CONDITIONS:
            continue
        grouped[str(row["example_id"])][condition].append(row)

    pairs: list[dict[str, Any]] = []
    for example_id, by_cond in grouped.items():
        n = min(len(by_cond["no_context"]), len(by_cond["fake_context"]))
        for pair_idx in range(n):
            no_row = by_cond["no_context"][pair_idx]
            fake_row = by_cond["fake_context"][pair_idx]
            pairs.append(
                {
                    "pair_id": f"{example_id}__{pair_idx}",
                    "example_id": example_id,
                    "pair_idx": pair_idx,
                    "question": no_row["question"],
                    "original_answer": no_row.get("original_answer"),
                    "substituted_answer": no_row.get("substituted_answer"),
                    "substitution_type": no_row.get("substitution_type"),
                    "no_context": no_row,
                    "fake_context": fake_row,
                }
            )
    return pairs


def condition_mean_std(
    by_layer: pd.DataFrame,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two layer-vs-value curves: the mean over no_context samples and the mean over fake_context samples."""
    n_layers = int(by_layer["layer"].max()) + 1
    layers = np.arange(n_layers)
    no_means, no_stds, fake_means, fake_stds = [], [], [], []
    for layer in layers:
        no_vals = by_layer.loc[
            (by_layer["condition"] == "no_context") & (by_layer["layer"] == layer),
            value_col,
        ].to_numpy(dtype=float)
        fake_vals = by_layer.loc[
            (by_layer["condition"] == "fake_context") & (by_layer["layer"] == layer),
            value_col,
        ].to_numpy(dtype=float)
        no_means.append(float(np.mean(no_vals)) if len(no_vals) else float("nan"))
        no_stds.append(float(np.std(no_vals)) if len(no_vals) else float("nan"))
        fake_means.append(float(np.mean(fake_vals)) if len(fake_vals) else float("nan"))
        fake_stds.append(float(np.std(fake_vals)) if len(fake_vals) else float("nan"))
    return (
        layers,
        np.asarray(no_means),
        np.asarray(no_stds),
        np.asarray(fake_means),
        np.asarray(fake_stds),
    )


def paired_delta_mean_std(
    by_layer: pd.DataFrame,
    value_col: str,
    delta_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± std of paired ``delta_sign * (fake - no)`` at each layer.

    With the current ``METRIC_SPECS`` (``delta_sign=-1``) this is ``no - fake``.
    """
    n_layers = int(by_layer["layer"].max()) + 1
    layers = np.arange(n_layers)
    means, stds = [], []
    for layer in layers:
        layer_df = by_layer[by_layer["layer"] == layer]
        no = layer_df[layer_df["condition"] == "no_context"].set_index("pair_id")[value_col]
        fake = layer_df[layer_df["condition"] == "fake_context"].set_index("pair_id")[
            value_col
        ]
        shared = no.index.intersection(fake.index)
        deltas = delta_sign * (
            fake.loc[shared].to_numpy(dtype=float) - no.loc[shared].to_numpy(dtype=float)
        )
        means.append(float(np.mean(deltas)) if len(deltas) else float("nan"))
        stds.append(float(np.std(deltas)) if len(deltas) else float("nan"))
    return layers, np.asarray(means), np.asarray(stds)


def auc_by_layer(
    by_layer: pd.DataFrame,
    value_col: str,
    positive_condition: str,
) -> pd.DataFrame:
    rows = []
    n_layers = int(by_layer["layer"].max()) + 1
    for layer in range(n_layers):
        layer_df = by_layer[by_layer["layer"] == layer]
        y = (layer_df["condition"] == positive_condition).astype(int).to_numpy()
        scores = layer_df[value_col].to_numpy(dtype=float)
        rows.append({"layer": layer, "auc": float(roc_auc_score(y, scores))})
    return pd.DataFrame(rows)
