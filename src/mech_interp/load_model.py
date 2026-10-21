"""TransformerLens model loading for NQ-Swap analyses."""

from __future__ import annotations

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm.llm_utils import get_model_id


def load_tl_model(
    model_name: str,
    device: str,
    attn_implementation: str = "eager",
    *,
    center_unembed: bool = False,
) -> HookedTransformer:
    """Load a HookedTransformer; HF path used for Llama/Gemma/Mistral/Qwen."""
    name = model_name.lower()
    if any(tag in name for tag in ("llama", "gemma", "mistral", "qwen")):
        model_id = get_model_id(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="bfloat16",
            device_map="cpu",
            attn_implementation=attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return HookedTransformer.from_pretrained(
            model_name=model_id,
            fold_ln=True,
            center_writing_weights=False,
            center_unembed=center_unembed,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
        )

    return HookedTransformer.from_pretrained(
        model_name,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
        dtype="bfloat16",
        attn_implementation=attn_implementation,
    )
