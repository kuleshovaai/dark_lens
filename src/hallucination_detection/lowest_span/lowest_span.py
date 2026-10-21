from dataclasses import dataclass
from typing import List, Iterable, Optional
import torch
import torch.nn.functional as F
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.llm import LLMOutputProvider, LLMConfig
from src.customtypes import LLMOutputs, LLMResponse



@dataclass
class LowestSpan(WhiteBoxHalDetMethod):
    llm_config: LLMConfig
    batch_size: int = 4
    window_size: int = 3
    use_cache: bool = True
    cache_dir: str = "cache/lowest_span"

    def __post_init__(self):
        """Post initialization of the class."""
        super().__init__(batch_size=self.batch_size, use_cache=self.use_cache, cache_dir=self.cache_dir)

    @property
    def config(self):
        config = {
            "llm_model": repr(self.llm_config),
            "llm_model_config": self.llm_config.config.to_dict(),
            "window_size": self.window_size,
        }
        return config

    def get_score(
        self,
        queries_batch: List[LLMResponse],
        outputs_batch: List[LLMOutputs],
    ) -> List[List[float]]:
        scores = []
        for queries, outputs in zip(queries_batch, outputs_batch):
            assert len(queries["response"]) == 1, "Multiple generations encountered in Perplexity"

            llm_forward_output, prompt_ids, answer_ids = outputs
            logits = llm_forward_output.logits

            answer_ids, attention_mask = answer_ids.input_ids, answer_ids.attention_mask
            prompt_len, response_len = prompt_ids.input_ids.shape[1], attention_mask.sum().item()
            if logits is None:
                raise ValueError("Expected to get `logits`, but got None.")

            response_logits = logits[:, prompt_len - 1 : -1]        # logits[n] are used to predict (n+1)-th token
            logprob = torch.log(torch.softmax(response_logits, dim=-1))
            answer_ids = answer_ids.unsqueeze(-1).to(logprob.device)
            logprob = torch.gather(logprob, dim=-1, index=answer_ids)
            logprob = logprob.squeeze(-1)[:, :response_len]      # [1, response_len]

            probs = torch.exp(logprob).reshape(1, 1, response_len)

            kernel = torch.ones(1, 1, self.window_size, device=probs.device, dtype=probs.dtype)

            if response_len < 3:
                window_sums = probs.sum()
            else:
                window_sums = F.conv1d(probs, kernel, padding=0)
            
            score = window_sums.min() / self.window_size
            scores.append([(1. - score).float().item()])

        return scores

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