from abc import ABC, abstractmethod
from typing import Dict, Any
from transformers import PreTrainedModel


class Wrapper(ABC):
    @abstractmethod
    def __init__(self, llm_model: PreTrainedModel):
        """Applies wrapper to llm_model."""
        pass

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """Collects internal state of wrapper."""
        pass

    @abstractmethod
    def enable(self):
        """Enables modified computational flow through `llm_model`."""
        pass

    @abstractmethod
    def disable(self):
        """Disables modified computational flow through `llm_model`."""
        pass

    @abstractmethod
    def restore(self):
        """Restores original state of `llm_model`."""
        pass
