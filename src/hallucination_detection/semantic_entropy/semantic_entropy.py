from dataclasses import dataclass
from typing import Dict, List, Iterable
import nltk
import torch
from .utils import (
    EntailmentDeberta,
    _get_semantic_ids,
    _calc_loss,
    _logsumexp_by_id,
    _predictive_entropy_rao,
)
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.customtypes import LLMResponse, LLMOutputs
from src.llm import LLMOutputProvider, LLMConfig


@dataclass
class SemanticEntropy(WhiteBoxHalDetMethod):
    llm_config: LLMConfig
    device: str = "cuda"
    batch_size: int = 4
    use_cache: bool = True
    cache_dir: str = "cache/semantic_entropy"

    def __post_init__(self):
        super().__init__(
            batch_size=self.batch_size,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
        )
        nltk.download("punkt")
        self.entailment_model = EntailmentDeberta(self.device)

    @property
    def config(self) -> Dict:
        config = {
            "llm_model": repr(self.llm_config),
            "llm_model_config": self.llm_config.config.to_dict(),
        }
        return config

    def get_score(
        self,
        queries_batch: List[LLMResponse],
        outputs_batch: List[LLMOutputs],
    ) -> List[List[float]]:
        scores = []
        for query, outputs in zip(queries_batch, outputs_batch):
            responses = query["response"]

            llm_forward_outputs, prompt_ids, answer_ids = outputs
            logits = llm_forward_outputs.logits
            if logits is None:
                raise ValueError("Expected to get `logits`, but got None.")
            sample_ids = torch.cat(
                (prompt_ids.input_ids, answer_ids.input_ids),
                dim=1
            )
            loss_mask = torch.cat(
                (torch.zeros_like(~prompt_ids.special_tokens_mask), answer_ids.attention_mask),
                dim=1
            ).bool()
            masked_ids = sample_ids.masked_fill(~loss_mask, -100).to(logits.device)

            loss = _calc_loss(logits, masked_ids).cpu().float()
            log_likelihood = [-elem for elem in loss]

            semantic_ids = _get_semantic_ids(
                responses, model=self.entailment_model, strict_entailment=False
            )

            log_likelihood_per_semantic_id = _logsumexp_by_id(
                semantic_ids, log_likelihood, agg="sum_normalized"
            )
            pe = _predictive_entropy_rao(log_likelihood_per_semantic_id)

            scores.append([float(pe)])
        return scores
    
    def predict_score(self, X: List[List[float]]) -> List[float]:
        return [x[0] for x in X]

    def fit_hyperparameters(
        self,
        stream: Iterable[LLMResponse],
        outputs_provider: LLMOutputProvider,
        y_val: List[int],
    ):
        """Perform hyperparameter tuning on the validation set if needed."""
        pass

    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: List[List[float]],
        y_val: List[int],
    ) -> "WhiteBoxHalDetMethod":
        """Fit the detection model on training data."""
        return self

    def reset(self):
        """For supervised models - return the model to the untrained state."""
        pass
