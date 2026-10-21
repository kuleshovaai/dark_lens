from typing import Any, cast
from transformers import PretrainedConfig
from .llm_stream import LLMOutputProvider
from .llm_base import LLMBase as LLM


def resolve_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the text tower config (Gemma3 nests it under ``text_config``)."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return cast(PretrainedConfig, text_config)
    return config


class LLMConfig:
    __slots__ = ("model_name", "config", "_repr")
    
    def __init__(self, llm: "LLM"):
        config = cast(PretrainedConfig, llm.config)
        self.config = config
        self._repr = repr(llm)
        self.model_name = llm.model_name

    def __repr__(self) -> str:
        return self._repr

    @property
    def text_config(self) -> PretrainedConfig:
        return resolve_text_config(self.config)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.text_config.num_hidden_layers)

    def text_config_to_dict(self) -> Any:
        tc = self.text_config
        if hasattr(tc, "to_dict"):
            return tc.to_dict()
        return dict(tc)
