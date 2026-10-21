from dataclasses import dataclass
from typing import List, Iterable, Optional
import torch
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.llm import LLMOutputProvider, LLMConfig
from src.customtypes import LLMOutputs, LLMResponse



@dataclass
class Perplexity(WhiteBoxHalDetMethod):
    llm_config: LLMConfig
    batch_size: int = 4
    min_k: Optional[float] = None
    use_cache: bool = True
    cache_dir: str = "cache/perplexity"

    def __post_init__(self):
        """Post initialization of the class."""
        super().__init__(batch_size=self.batch_size, use_cache=self.use_cache, cache_dir=self.cache_dir)

    @property
    def config(self):
        config = {
            "llm_model": repr(self.llm_config),
            "llm_model_config": self.llm_config.config.to_dict(),
            "min_k": self.min_k,
        }
        return config

    def get_score(
        self,
        queries_batch: List[LLMResponse],
        outputs_batch: List[LLMOutputs],
    ) -> List[List[float]]:
        perplexity_scores = []
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

            if self.min_k is not None:
                logprob = torch.topk(
                    logprob, 
                    k=max(int(self.min_k * logprob.shape[0]), 1), 
                    dim=-1, 
                    largest=False
                )[0]

            perplexity = torch.exp(-logprob.mean()).float().item()
            perplexity_scores.append([perplexity])

        return perplexity_scores

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
