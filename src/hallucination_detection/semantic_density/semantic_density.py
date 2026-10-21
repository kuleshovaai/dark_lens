from dataclasses import dataclass
from typing import Dict, List, Iterable
import re
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.customtypes import LLMResponse, LLMOutputs
from src.llm import LLMOutputProvider, LLMConfig


@dataclass
class SemanticDensity(WhiteBoxHalDetMethod):
    llm_config: LLMConfig
    device: str = "cuda"
    batch_size: int = 1
    use_cache: bool = True
    cache_dir: str = "cache/semantic_density"

    def __post_init__(self):
        super().__init__(
            batch_size=self.batch_size,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
        )
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        self.nli_tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", dtype="bfloat16").to(self.device)
        self.nli_model.eval()
        
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
            
            base_response = responses[0]
            unique_responses, unique_idx = np.unique(responses[1:], return_index=True)
            order = np.argsort(unique_idx)
            unique_responses = unique_responses[order]
            unique_idx = unique_idx[order]
            positions = [0] + (unique_idx + 1).tolist()

            llm_forward_outputs, prompt_ids, answer_ids = outputs
            logits = llm_forward_outputs.logits
            if logits is None:
                raise ValueError("Expected to get `logits`, but got None.")
            response_lengths = answer_ids.attention_mask[positions].sum(dim=1).numpy()
            prompt_length = prompt_ids.input_ids.shape[1]
            
            prompt = self.filter_prompt(query["prompt"], prompt_ids)
            semantic_distance = self.get_semantic_distance(prompt, base_response, unique_responses, reverse=False)
            reverse_semantic_distance = self.get_semantic_distance(prompt, base_response, unique_responses, reverse=True)
            semantic_density = ((1 - semantic_distance) + (1 - reverse_semantic_distance)) / 2
            semantic_density = [1] + semantic_density.tolist() # 1 is added for base_response
            
            response_logits = logits[positions, prompt_length - 1 : -1]        # logits[n] are used to predict (n+1)-th token
            logprobs = torch.log_softmax(response_logits, dim=-1)
            answer_ids = answer_ids.input_ids[positions].unsqueeze(-1).to(logprobs.device)
            logprobs = torch.gather(logprobs, dim=-1, index=answer_ids)
            logprobs = logprobs.squeeze(-1).cpu().float()
            
            if "gemma" in self.llm_config.model_name:
                length_penalty = self.llm_config.config.text_config.length_penalty
            else:
                length_penalty = self.llm_config.config.length_penalty
            
            probs_list = []
            for logprob, length in zip(logprobs, response_lengths):
                probs_list.append(torch.exp(logprob[:length].sum() / length ** length_penalty).item())
            assert len(probs_list) == len(semantic_density)
            norm = sum(probs_list)
            score = sum([p * d / norm for p, d in zip(probs_list, semantic_density)])
            scores.append([1 - score])
        return scores
    
    @staticmethod
    def filter_prompt(text, tokens):
        filtered_text = []
        for mask_val, mapping in zip(tokens["special_tokens_mask"][0], tokens["offset_mapping"][0]):
            if mask_val:
                continue
            filtered_text.append(text[mapping[0]: mapping[1]])
        return "".join(filtered_text)
    
    @torch.no_grad()
    def get_semantic_distance(self, question, target, samples, reverse=False):
        entail_id = self.nli_model.deberta.config.label2id["entailment"]  
        contra_id = self.nli_model.deberta.config.label2id["contradiction"]
        neutral_id = self.nli_model.deberta.config.label2id["neutral"]
        
        if reverse:
            text_pairs = [
                [sample, question + target]
                for sample in samples
            ]
        else:
            text_pairs = [
                [question + target, sample]
                for sample in samples
            ]

        probs = []
        dataloader = torch.utils.data.DataLoader(text_pairs, batch_size=self.batch_size)
        for batch in dataloader:
            text_pairs_tokenized = self.nli_tokenizer.batch_encode_plus(
                list(zip(*batch)),
                padding=True,
                return_tensors="pt",
            ).to(self.nli_model.device)
            logit = self.nli_model(**text_pairs_tokenized).logits
            prob = F.softmax(logit, dim=-1).cpu()
            probs.append(prob)
        probs = torch.cat(probs, dim=0)
        semantic_distance = probs[:, contra_id] + 0.5 * probs[:, neutral_id]
        semantic_distance = semantic_distance.clamp(max=1.0)
        return semantic_distance
    
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
