from dataclasses import dataclass
from typing import Iterable, List, Literal
import nltk
import torch
import numpy as np
from src.customtypes import LLMOutputs, LLMResponse
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.llm import LLMConfig, LLMOutputProvider
from sentence_transformers.cross_encoder import CrossEncoder


@dataclass
class SAR(WhiteBoxHalDetMethod):
    llm_config: LLMConfig
    method: Literal["token_sar", "sentence_sar", "sar"] = "sar"
    alpha: float = 0.5
    batch_size: int = 1
    device: str = "cuda"
    use_cache: bool = True
    cache_dir: str = "cache/sar"

    def __post_init__(self):
        super().__init__(
            batch_size=self.batch_size,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
        )
        nltk.download('punkt')
        self.nli_model = CrossEncoder("cross-encoder/stsb-roberta-large", num_labels=1, device=self.device)

    @property
    def config(self):
        return {
            "llm_model": repr(self.llm_config),
            "llm_model_config": self.llm_config.config.to_dict(),
            "alpha": self.alpha,
        }
    
    @staticmethod
    def filter_prompt(text, tokens):
        filtered_text = []
        for mask_val, mapping in zip(tokens["special_tokens_mask"][0], tokens["offset_mapping"][0]):
            if mask_val:
                continue
            filtered_text.append(text[mapping[0]: mapping[1]])
        return "".join(filtered_text)

    def token_importance(self, prompt, response_batch, list_offset_mapping):
        token_importances = []
        for response_sent, offset_mapping in zip(response_batch, list_offset_mapping):
            sent_start = offset_mapping[0, 0]
            premises = []
            hypotheses = []
            for token_pos in offset_mapping:
                local_start = token_pos[0] - sent_start
                local_end   = token_pos[1] - sent_start
                premises.append(prompt + response_sent)
                hypotheses.append(
                    prompt
                    + response_sent[:local_start]
                    + response_sent[local_end:]
                )
            similarities = self.nli_model.predict(
                list(zip(premises, hypotheses)),
                show_progress_bar=False,
            )
            token_importance = 1 - torch.tensor(similarities)
            token_importances.append(token_importance)
        return token_importances

    @staticmethod
    def token_sar(token_importances, list_probs):
        scores = []
        for importance, prob in zip(token_importances, list_probs):
            token_wise_entropy = -torch.log(prob)
            weighted_score = ((importance / importance.sum()) * token_wise_entropy)
            scores.append(weighted_score.sum())
        scores = torch.stack(scores)
        return scores
    
    def sentence_similarity(self, prompt, response_sentences):
        num_sent = len(response_sentences)
        if num_sent == 1:
            return torch.tensor([[1.0]])
        pairs = []
        premises = []
        hypotheses = []
        for i in range(num_sent):
            for j in range(i + 1, num_sent):
                pairs.append((i, j))
                premises.append(prompt + response_sentences[i])
                hypotheses.append(prompt + response_sentences[j])
        scores = self.nli_model.predict(
            list(zip(premises, hypotheses)),
            show_progress_bar=False,
        )
        similarities = np.eye(num_sent)
        for (i, j), score in zip(pairs, scores):
            similarities[i, j] = score
            similarities[j, i] = score
        return torch.from_numpy(similarities)
        
    def get_score(
        self,
        queries_batch: List[LLMResponse],
        outputs_batch: List[LLMOutputs],
    ) -> List[List[float]]:
        scores = []
        for queries, outputs in zip(queries_batch, outputs_batch):
            llm_forward_output, prompt_ids, answer_ids = outputs
            logits = llm_forward_output.logits
            
            prompt_len = prompt_ids.input_ids.shape[1]
            
            if logits is None:
                raise ValueError("Expected to get `logits`, but got None.")
            response_logits = logits[:, prompt_len - 1: -1]

            batch_probs = torch.softmax(response_logits, dim=-1)
            batch_probs = torch.gather(
                batch_probs,
                dim=-1,
                index=answer_ids.input_ids[..., None].to(batch_probs.device),
            ).squeeze(-1).cpu()

            list_probs = [p[m == 1][: m.sum()] for p, m in zip(batch_probs, answer_ids.attention_mask)]
            list_offset_mapping = [o[: m.sum()] for o, m in zip(answer_ids.offset_mapping, answer_ids.attention_mask)]

            prompt = self.filter_prompt(queries["prompt"], prompt_ids)
            
            token_importances = self.token_importance(prompt, queries["response"], list_offset_mapping)
            sent_probs = torch.stack([torch.prod(prob) for prob in list_probs])
            sent_similarities = self.sentence_similarity(prompt, queries["response"])
            
            token_sar = self.token_sar(token_importances, list_probs)
            exp_token_sar = torch.exp(-token_sar)

            sent_sar = torch.tensor(
                [
                    -torch.log(
                        sent_probs[i] + self.alpha * (
                            (sent_probs[:i] * sent_similarities[i, :i]).sum() + \
                            (sent_probs[i + 1:] * sent_similarities[i, i + 1:]).sum()
                        )
                    ) for i in range(len(sent_probs))
                ]
            )
            sar = torch.tensor(
                [
                    -torch.log(
                        exp_token_sar[i] + self.alpha * (
                            (exp_token_sar[:i] * sent_similarities[i, :i]).sum() + \
                            (exp_token_sar[i + 1:] * sent_similarities[i, i + 1:]).sum()
                        )
                    ) for i in range(len(exp_token_sar))
                ]
            )
            scores.append([token_sar.mean(), sent_sar.mean(), sar.mean()])
        return scores
    
    def predict_score(self, X: List[List[float]]) -> List[float]:
        if self.method == "token_sar":
            return [x[0] for x in X]
        if self.method == "sentence_sar":
            return [x[1] for x in X]
        if self.method == "sar":
            return [x[2] for x in X]
        raise NotImplementedError
    
    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: List[List[float]],
        y_val: List[int],
    ) -> "SAR":
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
