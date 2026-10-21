from dataclasses import dataclass
from typing import List, Iterable, Optional
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.llm import LLMOutputProvider, LLMConfig
from src.customtypes import LLMOutputs, LLMResponse


@dataclass
class LLMCheck(WhiteBoxHalDetMethod):
    llm_config: LLMConfig
    batch_size: int = 4
    use_cache: bool = True
    cache_dir: str = "cache/llm_check"
    best_layer: Optional[int] = None

    def __post_init__(self):
        """Post initialization of the class."""
        super().__init__(
            batch_size=self.batch_size,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
        )

    @property
    def config(self):
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
        attention_scores = []
        for queries, outputs in zip(queries_batch, outputs_batch):
            assert len(queries["response"]) == 1, "Multiple generations encountered in LLM-Check"
            llm_forward_output, prompt_ids, answer_ids = outputs
            attention_maps = llm_forward_output.attentions
            if attention_maps is None:
                raise ValueError(f"Expected to get `attentions`, but got None.")
            prompt_padding = (1 - prompt_ids.attention_mask).sum().item()
            sample_len = (prompt_ids.attention_mask.sum() + answer_ids.attention_mask.sum()).item()
            layer_scores = [
                self._logdet(
                    layer_maps[
                        0, 
                        :,
                        prompt_padding : prompt_padding + sample_len,
                        prompt_padding : prompt_padding + sample_len,
                    ]
                ) for layer_maps in attention_maps
            ]
            attention_scores.append(layer_scores)
        return attention_scores

    def predict_score(self, X: List[List[float]]) -> List[float]:
        if self.best_layer is None:
            raise ValueError(f"Expected to get `self.best_layer`, but got None.")
        return [x[self.best_layer] for x in X]

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
        features_batch = [
            np.array(features) for features in self.transform(
                stream=stream,
                outputs_provider=outputs_provider,
            )
        ]

        features_batch = np.concatenate(features_batch)

        num_layers = self.llm_config.config.num_hidden_layers

        max_roc_auc, best_idx = 0, None
        for layer_idx in range(num_layers):
            features = features_batch[:, layer_idx]
            roc_auc = roc_auc_score(y_val, features)
            if roc_auc > max_roc_auc:
                best_idx = layer_idx
                max_roc_auc = roc_auc

        self.best_layer = best_idx

    def _fix_zeros(self, attention_map: torch.Tensor) -> torch.Tensor:
        attention_map[attention_map < 1e-5] = 1e-5
        return attention_map

    def _logdet(self, attention_map: torch.Tensor) -> float:
        """Calculate attention score for an attention map

        Args:
            attention_map (torch.Tensor): tensor of shape (n_heads, n_tokens, n_tokens).

        Returns:
            float: attention score.

        """
        return torch.log(
                torch.diagonal(self._fix_zeros(attention_map.float()), dim1=-2, dim2=-1)
            ).mean().item()
