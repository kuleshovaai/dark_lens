from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import torch
from dataclasses import dataclass
from src.customtypes import LLMResponse, LLMOutputs
from src.llm import LLMOutputProvider, LLMConfig
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from .utils import token_level_focus_scores, load_idf


@dataclass
class Focus(WhiteBoxHalDetMethod):
    llm_config: LLMConfig
    
    gamma: float
    p: float
    
    idf_dataset: str
    seed: int
    idf_dataset_size: int
    spacy_path: str
    
    batch_size: int = 1
    use_cache: bool = True
    cache_dir: str = "cache/focus"

    def __post_init__(self):
        super().__init__(batch_size=self.batch_size, use_cache=self.use_cache, cache_dir=self.cache_dir)
        self.idf_stats = load_idf(
            self.llm_config.model_name,
            f"{Path(self.cache_dir).parts[0]}/idf_{self.llm_config.model_name}.pkl",
            self.idf_dataset,
            trust_remote_code=True,
            idf_seed=self.seed,
            idf_dataset_size=self.idf_dataset_size,
            spacy_path=self.spacy_path,
        )

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "llm_model": repr(self.llm_config),
            "llm_model_config": self.llm_config.config.to_dict(),
            "p": self.p,
            "gamma": self.gamma,
        }

    def reset(self):
        self.best_thr = None

    def get_score(
        self,
        queries_batch: List[LLMResponse],
        outputs_batch: List[LLMOutputs],
    ) -> List[List[float]]:
        scores = []
        for queries, outputs in zip(queries_batch, outputs_batch):
            assert len(queries["response"]) == 1, "Multiple generations encountered in RAUQ"

            llm_forward_output, prompt_ids, answer_ids = outputs
            logits = llm_forward_output.logits
            attentions = llm_forward_output.attentions

            prompt_len, response_len = prompt_ids.input_ids.shape[1], answer_ids.attention_mask.sum().item()
            if logits is None:
                raise ValueError("Expected to get `logits`, but got None.")
            if attentions is None:
                raise ValueError("Expected to get `attentions`, but got None.")

            log_probs = torch.nn.functional.log_softmax(logits[0, prompt_len:], dim=-1)
            max_token_attn = tuple([
                attn[
                    0,
                    :,
                    prompt_len: prompt_len + response_len,
                    prompt_len: prompt_len + response_len,
                ].max(dim=0).values.float().cpu() for attn in attentions
            ])
            max_token_attn = torch.stack(max_token_attn, dim=0).max(dim=0).values
            token_focus, kw_mask = token_level_focus_scores(
                queries["response"][0],
                answer_ids.input_ids[0, :response_len],
                answer_ids.offset_mapping[0, :response_len - 1],
                log_probs.cpu().float(),
                max_token_attn,
                self.idf_stats,
                self.p,
                self.gamma,
            )
            token_focus_np = np.array(token_focus)
            if token_focus_np.size == 0 or kw_mask.sum() == 0:
                score = 0
            else:
                score = np.mean(token_focus_np[kw_mask[: len(token_focus_np)]])
            scores.append([float(score)])
                
        return scores

    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: List[List[float]],
        y_val: List[int],
    ):
        return self

    def predict_score(self, X: List[List[float]]) -> List[float]:
        return [max(sample_scores) for sample_scores in X]

    def fit_hyperparameters(
        self,
        stream,
        outputs_provider: LLMOutputProvider,
        y_val: List[int],
    ):
        pass
