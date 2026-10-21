from dataclasses import dataclass
from typing import Literal, List, Iterable
import torch
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.llm import LLMOutputProvider, LLMConfig
from src.customtypes import LLMOutputs, LLMResponse
from src.batching_utils import concat_outputs


@dataclass
class TokenwiseEntropy(WhiteBoxHalDetMethod):
    llm_config: LLMConfig
    batch_size: int = 4
    aggregation: Literal["max", "min", "mean"] = "mean"
    use_cache: bool = True
    cache_dir: str = "cache/tokenwise_entropy"

    def __post_init__(self):
        """Post initialization of the class."""
        super().__init__(batch_size=self.batch_size, use_cache=self.use_cache, cache_dir=self.cache_dir)

    @property
    def config(self):
        config = {
            "llm_model": repr(self.llm_config),
            "llm_model_config": self.llm_config.config.to_dict(),
            "aggregation": self.aggregation,
        }
        return config
    
    @staticmethod
    def aggregate(
        X: torch.Tensor,
        mask: torch.Tensor,
        aggregation: Literal["mean", "min", "max"] = "mean",
    ) -> torch.Tensor:
        mask = mask.bool().to(X.device)
        if aggregation == "mean":
            masked = X * mask
            return masked.sum(dim=1) / mask.sum(dim=1)
        if aggregation == "min":
            masked = X.masked_fill(~mask, float("inf"))
            return masked.min(dim=1).values
        if aggregation == "max":
            masked = X.masked_fill(~mask, float("-inf"))
            return masked.max(dim=1).values
        raise ValueError(f"Unsupported aggregation method: {aggregation}")

    def get_score(
        self,
        queries_batch: List[LLMResponse],
        outputs_batch: List[LLMOutputs],
    ) -> List[List[float]]:
        outputs_concatenated = concat_outputs(
            *outputs_batch,
            pad_token_id=0,
        )
        llm_forward_output, prompt_ids, answer_ids = outputs_concatenated
        logits = llm_forward_output.logits
        if logits is None:
            raise ValueError("Expected to get `logits`, but got None.")
        
        mask_left = torch.zeros_like(prompt_ids.attention_mask)
        mask_left[:, -1] = 1
        mask_right = answer_ids.attention_mask.clone()
        # Drop EOS (last non-pad answer token); pads are already 0 in attention_mask.
        last_answer_idx = mask_right.long().sum(dim=1) - 1
        valid = last_answer_idx >= 0
        batch_idx = torch.arange(mask_right.size(0), device=mask_right.device)
        mask_right[batch_idx[valid], last_answer_idx[valid]] = 0

        mask = torch.cat(
            (mask_left, mask_right),
            dim=1
        ).bool().to(logits.device)
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log(probabilities + 1e-6)
        token_entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
 
        entropies = self.aggregate(token_entropy, mask, self.aggregation)
        return entropies.unsqueeze(1).cpu().tolist()

    def predict_score(self, X: List[List[float]]) -> List[float]:
        return [x[0] for x in X]
    
    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: List[List[float]],
        y_val: List[int],
    ):
        return self

    def reset(self):
        """For supervised models - return the model to the untrained state."""
        pass

    def fit_hyperparameters(
        self,
        stream: Iterable[LLMResponse],
        outputs_provider: LLMOutputProvider,
        y_val: List[int],
    ):
        pass
