from dataclasses import dataclass
from typing import Iterable, List
import torch
from src.customtypes import LLMOutputs, LLMResponse
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.llm import LLMConfig, LLMOutputProvider


@dataclass
class RAUQ(WhiteBoxHalDetMethod):
    llm_config: LLMConfig
    alpha: float = 0.5
    batch_size: int = 1
    use_cache: bool = True
    cache_dir: str = "cache/rauq"

    def __post_init__(self):
        super().__init__(
            batch_size=self.batch_size,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
        )

    @property
    def config(self):
        return {
            "llm_model": repr(self.llm_config),
            "llm_model_config": self.llm_config.config.to_dict(),
            "alpha": self.alpha,
        }

    def get_score(
        self,
        queries_batch: List[LLMResponse],
        outputs_batch: List[LLMOutputs],
    ) -> List[List[float]]:
        rauq_scores = []
        for queries, outputs in zip(queries_batch, outputs_batch):
            assert len(queries["response"]) == 1, "Multiple generations encountered in RAUQ"

            llm_forward_output, prompt_ids, answer_ids = outputs
            logits = llm_forward_output.logits
            attentions = llm_forward_output.attentions
            answer_ids, attention_mask = answer_ids.input_ids, answer_ids.attention_mask
            answer_ids = answer_ids.to(logits.device)
            prompt_len, response_len = prompt_ids.input_ids.shape[1], attention_mask.sum().item()
            if logits is None:
                raise ValueError("Expected to get `logits`, but got None.")
            if attentions is None:
                raise ValueError("Expected to get `attentions`, but got None.")
            attn_heads = tuple([
                attn[
                    0,
                    :,
                    prompt_len: prompt_len + response_len,
                    prompt_len: prompt_len + response_len,
                ] for attn in attentions
            ])
            
            attn_to_prev = []
            for attn_l in attn_heads:
                scores = torch.diagonal(attn_l, offset=-1, dim1=1, dim2=2)
                head_idx = torch.argmax(scores.mean(dim=-1))
                attn_to_prev.append(scores[head_idx].cpu().float())
            attn_to_prev = torch.stack(attn_to_prev, dim=0)
            
            response_logits = logits[0, prompt_len: prompt_len + response_len]
            probs = torch.softmax(response_logits, dim=-1)
            probs = torch.gather(probs, dim=-1, index=answer_ids[:, :response_len]).cpu()
            probs = probs.repeat(attn_to_prev.shape[0], 1)
            
            c_prev = probs[:, 0]
            c_values = [c_prev]
            for i in range(1, probs.shape[1]):
                c = (
                    self.alpha * probs[:, i]
                    + (1.0 - self.alpha) * attn_to_prev[:, i - 1] * c_prev
                )
                c_prev = c
                c_values.append(c)
            c_tensor = torch.stack(c_values, dim=0)
            u = -torch.log(c_tensor).mean(dim=-1)
            rauq_scores.append([u.max().item()])
        return rauq_scores
    
    def predict_score(self, X: List[List[float]]) -> List[float]:
        return [x[0] for x in X]
    
    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: List[List[float]],
        y_val: List[int],
    ) -> "RAUQ":
        return self

    def reset(self):
        pass

    def fit_hyperparameters(
        self,
        stream: Iterable[LLMResponse],
        outputs_provider: LLMOutputProvider,
        y_val: List[int],
    ):
        pass
