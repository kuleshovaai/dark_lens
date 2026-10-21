from dataclasses import dataclass
from typing import Iterable, List, Literal
from typing_extensions import override
import numpy as np
import torch
from torch.nn import functional as F
from src.customtypes import LLMOutputs, LLMResponse
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.llm import LLMConfig, LLMOutputProvider

@dataclass
class LUMINA(WhiteBoxHalDetMethod):
    """LUMINA detector combining internal predictive ratios with context MMD.

    Requires `WrapperForAfterLayerLogits`. Scores each response with (1) an
    IPR signal from after-layer entropy/greedy probs versus the final
    distribution, and (2) an embedding-space MMD between the original and an
    obscured-context predictive distribution, then blends them with ``lam``.
    """
    llm_config: LLMConfig
    batch_size: int = 2
    use_cache: bool = True
    cache_dir: str = "cache/lumina"
    kernel: Literal["cosine", "rbf"] = "cosine" 
    lam: float = 0.5

    def __post_init__(self):
        super().__init__(batch_size=self.batch_size, use_cache=self.use_cache, cache_dir=self.cache_dir)
        
    @property
    def config(self):
        config = {
           "llm_model": repr(self.llm_config),
           "llm_model_config": self.llm_config.config.to_dict(),
           "kernel": self.kernel,
        }
        return config
    
    @torch.no_grad()
    def _compute_ipr(
        self,
        after_layer_entropy: torch.Tensor,
        after_layer_greedy_probs: torch.Tensor,
        final_greedy_probs: torch.Tensor,
        final_token_probs: torch.Tensor,
        eps: float = 1e-8,
    ):
        num_layers = self.llm_config.num_hidden_layers
        weights = 1.0 / (after_layer_entropy + eps)
        layer_indices = torch.arange(1, num_layers + 1, device=after_layer_entropy.device).unsqueeze(1)
        ratios = 1 - torch.clamp(after_layer_greedy_probs / final_greedy_probs.unsqueeze(0), max=1.0)
        weighted_ratios = ratios * layer_indices
        total_weighted_ratio = weighted_ratios.sum(dim=0)
        total_weight = (layer_indices * weights).sum(dim=0)
        ipr = (total_weighted_ratio / total_weight) * (final_token_probs / final_greedy_probs)
        return ipr.cpu()
    
    @torch.no_grad()
    def _get_topk_embeddings_and_probs(self, probs, k):
        topk_probs, topk_ids = torch.topk(probs, k, dim=-1)
        embeddings = self._embedding_layer(topk_ids)
        return topk_probs.float(), embeddings.float()
    
    @staticmethod
    def _rbf_kernel(x, y, sigma=1.0):
        # (T, k, d)
        diff = x[:, :, None, :] - y[:, None, :, :]   # (T, k, k, d)
        dist_sq = (diff ** 2).sum(dim=-1)            # (T, k, k)
        return torch.exp(-dist_sq / (2 * sigma ** 2)).float()
    
    @staticmethod
    def _cosine_kernel(x, y, eps=1e-8):
        x = F.normalize(x, p=2, dim=-1, eps=eps)
        y = F.normalize(y, p=2, dim=-1, eps=eps)
        # (T, k, d) @ (T, d, k) -> (T, k, k)
        return (1 + torch.matmul(x, y.transpose(-1, -2))) / 2
    
    @torch.no_grad()
    def _compute_mmd(self, p_prob, q_prob, k=100):
        p_top_k, p_embed = self._get_topk_embeddings_and_probs(p_prob, k)
        q_top_k, q_embed = self._get_topk_embeddings_and_probs(q_prob, k)
        if self.kernel == "cosine":
            K_pp = self._cosine_kernel(p_embed, p_embed)
            K_qq = self._cosine_kernel(q_embed, q_embed)
            K_pq = self._cosine_kernel(p_embed, q_embed)
        elif self.kernel == "rbf":
            K_pp = self._rbf_kernel(p_embed, p_embed)
            K_qq = self._rbf_kernel(q_embed, q_embed)
            K_pq = self._rbf_kernel(p_embed, q_embed)
        else:
            raise ValueError("Invalid kernel. Choose 'cosine' or 'rbf'.")
        mmd = (
            torch.einsum("ti,tij,tj->t", p_top_k, K_pp, p_top_k)
            + torch.einsum("ti,tij,tj->t", q_top_k, K_qq, q_top_k)
            - 2 * torch.einsum("ti,tij,tj->t", p_top_k, K_pq, q_top_k)
        )
        return mmd.cpu()
    
    def get_score(self, queries_batch: List[LLMResponse], outputs_batch: List[LLMOutputs]) -> List[List[float]]:
        scores = []
        for idx in range(0, len(outputs_batch), 2):
            outputs = outputs_batch[idx]
            llm_forward_output, prompt_ids, answer_ids = outputs
            token_ids, attention_mask = answer_ids.input_ids, answer_ids.attention_mask
            prompt_len = prompt_ids.input_ids.shape[1]
            response_len = int(attention_mask.sum().item())
            logits = llm_forward_output.logits[
                0,
                prompt_len - 1: prompt_len + response_len - 1,
            ]
            probs = F.softmax(logits, dim=-1)
            final_greedy_probs = torch.max(probs, dim=-1).values
            gathered_ids = token_ids[0, :response_len].unsqueeze(-1).to(probs.device)
            final_token_probs = torch.gather(probs, dim=-1, index=gathered_ids).squeeze(-1)

            after_layer_entropy = torch.stack([
                layer_entropy[
                    0, 
                    prompt_len - 1: prompt_len + response_len - 1,
                ] for layer_entropy in llm_forward_output["after_layer_entropy"]
            ], dim=0)
            after_layer_greedy_probs = torch.stack([
                layer_probs[
                    0, 
                    prompt_len - 1: prompt_len + response_len - 1,
                ] for layer_probs in llm_forward_output["after_layer_greedy_probs"]
            ], dim=0)

            ipr = self._compute_ipr(
                after_layer_entropy,
                after_layer_greedy_probs,
                final_greedy_probs,
                final_token_probs,
            )

            obscured_output, obscured_prompt, obscured_answer = outputs_batch[idx + 1]
            obscured_prompt_len = obscured_prompt.input_ids.shape[1]
            obscured_response_len = int(obscured_answer.attention_mask.sum().item())
            logits_with_random_context = obscured_output.logits[
                0,
                obscured_prompt_len - 1: obscured_prompt_len + obscured_response_len - 1,
            ]
            probs_with_random_context = F.softmax(logits_with_random_context, dim=-1)
            mmd = self._compute_mmd(probs.float(), probs_with_random_context.float(), k=100)
            scores.append([ipr.mean().item(), mmd.mean().item()])    
        return scores
    
    @override
    def transform(
        self,
        stream: Iterable[LLMResponse],
        outputs_provider: LLMOutputProvider,
    ) -> Iterable[List[List[float]]]:
        self._embedding_layer = outputs_provider.llm.model.get_input_embeddings()
        def stream_with_obscured_samples(stream):
            stream = iter(stream)
            try:
                first_query = next(stream)
            except StopIteration:
                return
            prev_query = first_query
            for current_query in stream:
                yield prev_query
                obscured_query = LLMResponse(prompt=current_query["prompt"], response=prev_query["response"])
                yield obscured_query
                prev_query = current_query
            yield prev_query
            obscured_query = LLMResponse(prompt=first_query["prompt"], response=prev_query["response"])
            yield obscured_query

        outputs_stream = outputs_provider(stream_with_obscured_samples(stream))
        queries_batch = []
        outputs_batch = []
        for outputs in outputs_stream:
            sample_ids = outputs.sample_ids
            num_seqs_per_sample = outputs.num_seqs
            assert all(map(lambda s: s == 1, num_seqs_per_sample)), "Expected single response per sample."
            num_samples = len(sample_ids)
            if self.batch_size < 2 or self.batch_size % 2:
                raise ValueError(
                    "LUMINA implementation requires the batch size to be even and greater than or equal to 2. "
                    "Subsequent pairs of samples are processed together when computing "
                    "the external context score, where the latter sample in each pair "
                    "corresponds to the obscured sample in the modified stream."
                )
            for i in range(num_samples):
                query = LLMResponse(
                    prompt=outputs[i].prompts[0],
                    response=outputs[i].responses,
                )
                queries_batch.append(query)
                llm_outputs = (
                    outputs[i].outputs,
                    outputs[i].prompt_tokens,
                    outputs[i].response_tokens,
                )
                outputs_batch.append(llm_outputs)
                if len(queries_batch) == self.batch_size:
                    yield self.get_score(queries_batch, outputs_batch)
                    queries_batch = []
                    outputs_batch = []
        if len(queries_batch) > 0:
            yield self.get_score(queries_batch, outputs_batch)

    def predict_score(self, X: List[List[float]]) -> List[float]:
        """Perform inference.

        Parameters
        ----------
        X : List[List[float]]
            List of lists of MTopDiv values from fixed model heads for each sample in the dataset.

        Returns
        -------
        List[float]
            List of predicted probabilities for each input example.

        """
        X_np = np.array(X)
        scores = self.lam * X_np[:, 0] - (1 - self.lam) * X_np[:, 1]
        return scores.tolist()
    
    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: List[List[float]],
        y_val: List[int],
    ) -> "LUMINA":
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