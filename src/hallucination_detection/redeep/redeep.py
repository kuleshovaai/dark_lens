from loguru import logger
from typing import List, Optional, Tuple, Iterable
import optuna
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score
from src.customtypes import LLMResponse, LLMOutputs
from src.llm import LLMConfig, LLMOutputProvider
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod


def _minmax(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def _column_auroc(y: List[int], scores: np.ndarray, invert: bool) -> List[float]:
    labels = np.asarray(y)
    aucs = []
    for column in scores.T:
        if np.unique(column).size < 2 or np.unique(labels).size < 2:
            aucs.append(0.5)
            continue
        pred = -column if invert else column
        aucs.append(float(roc_auc_score(labels, pred)))
    return aucs


def _sum_bounds(
    knowledge_diff_batch: List[np.ndarray],
    external_similarity_batch: List[np.ndarray],
    top_k: int,
    top_n: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[float, float], Tuple[float, float]]:
    kd_sums = [kd[:, :top_k].sum(axis=-1) for kd in knowledge_diff_batch]
    es_sums = [es[:, :top_n].sum(axis=-1) for es in external_similarity_batch]
    kd_min = min(float(s.min()) for s in kd_sums)
    kd_max = max(float(s.max()) for s in kd_sums)
    es_min = min(float(s.min()) for s in es_sums)
    es_max = max(float(s.max()) for s in es_sums)
    return kd_sums, es_sums, (kd_min, kd_max), (es_min, es_max)


@dataclass
class ReDeEP(WhiteBoxHalDetMethod):
    """Token-level ReDeEP: H = α · PKS − β · ECS after per-token sum and min-max scaling."""

    llm_config: LLMConfig
    copy_heads: List[Tuple[int, int]]
    batch_size: int = 1
    use_cache: bool = True
    cache_dir: str = "cache/redeep"

    knowledge_ffns: Optional[List[int]] = None

    top_k: Optional[int] = None
    top_n: Optional[int] = None
    alpha: Optional[float] = 1.0
    beta: Optional[float] = 0.2
    n_trials: int = 1000

    external_similarity_min_max_scale: Tuple[float, float] = (0.0, 1.0)
    knowledge_diff_min_max_scale: Tuple[float, float] = (0.0, 1.0)

    need_fit: bool = True
    chunk_size: int = 8

    _ranked_head_idx: List[int] = field(init=False, repr=False)
    _ranked_layer_idx: List[int] = field(init=False, repr=False)

    def __post_init__(self):
        super().__init__(batch_size=self.batch_size, use_cache=self.use_cache, cache_dir=self.cache_dir)
        if not self.copy_heads:
            raise ValueError("copy_heads is required. Resolve them with find_copy_heads before constructing ReDeEP.")
        self.copy_heads = sorted((int(layer), int(head)) for layer, head in self.copy_heads)
        if self.knowledge_ffns is None:
            self.knowledge_ffns = list(range(self.llm_config.config.num_hidden_layers))
        else:
            self.knowledge_ffns = sorted(int(layer) for layer in self.knowledge_ffns)
        if self.top_n is None:
            self.top_n = len(self.copy_heads)
        if self.top_k is None:
            self.top_k = len(self.knowledge_ffns)
        if self.alpha is None:
            self.alpha = 1.0
        if self.beta is None:
            self.beta = 0.2
        self._ranked_head_idx = list(range(len(self.copy_heads)))
        self._ranked_layer_idx = list(range(len(self.knowledge_ffns)))

    @property
    def config(self):
        return {
            "llm_model": repr(self.llm_config),
            "llm_model_config": self.llm_config.config.to_dict(),
            "copy_heads": self.copy_heads,
            "knowledge_ffns": self.knowledge_ffns,
            "chunk_size": self.chunk_size,
        }

    @staticmethod
    def _get_attn_ids_dict(heads):
        heads = sorted((int(layer), int(head)) for layer, head in heads)
        attns_ids_dict = dict()
        layers = sorted({layer for layer, _ in heads})
        for layer in layers:
            attns_ids_dict[layer] = sorted(head for l, head in heads if l == layer)
        return attns_ids_dict

    def _generation_positions(self, prompt_len: int, response_len: int, device) -> torch.Tensor:
        # Authors score the hidden state used to generate each answer token:
        # last prompt token through the second-to-last answer token.
        start = prompt_len - 1
        return torch.arange(start, start + response_len, device=device)

    def _get_external_similarity_and_knowledge_diff_for_sample(
        self,
        outputs: LLMOutputs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        copy_heads_dict = self._get_attn_ids_dict(self.copy_heads)
        if self.knowledge_ffns is None:
            raise ValueError("Expected `self.knowledge_ffns`, but got None.")
        knowledge_ffns = self.knowledge_ffns

        llm_forward_output, prompt_ids, answer_ids = outputs
        attn_heads = tuple(
            llm_forward_output["attentions"][layer_idx][0, heads_idxs]
            for layer_idx, heads_idxs in copy_heads_dict.items()
        )
        hidden_states = llm_forward_output.hidden_states
        if hidden_states is None:
            raise ValueError("Expected to get `hidden_states`, but got None.")
        divergences = llm_forward_output.divergences
        if divergences is None:
            raise ValueError("Expected to get `divergences`, but got None.")
        answer_ids, attention_mask = answer_ids.input_ids, answer_ids.attention_mask
        prompt_len = prompt_ids.input_ids.shape[1]
        response_len = int(attention_mask.sum().item())
        prompt_padding_len = int((~prompt_ids.attention_mask.bool()).sum().item())
        device = hidden_states[-1].device
        token_positions = self._generation_positions(prompt_len, response_len, device)

        if len(attn_heads) > 0:
            attn_heads = torch.cat(attn_heads, dim=0)
            last_hidden_layer = hidden_states[-1][0]
            num_heads = attn_heads.shape[0]
            k = max(1, int((prompt_len - prompt_padding_len) * 0.1))
            external_similarity = []
            for start in range(0, response_len, self.chunk_size):
                end = min(start + self.chunk_size, response_len)
                chunk_positions = token_positions[start:end]
                pointer_probs = attn_heads[:, chunk_positions, prompt_padding_len:prompt_len]
                top_k_indices = torch.topk(pointer_probs, k, dim=-1).indices
                top_k_hidden_states = F.embedding(top_k_indices, last_hidden_layer[prompt_padding_len:])
                attend_token_hidden_state = top_k_hidden_states.mean(dim=2)
                current_hidden_states = last_hidden_layer[chunk_positions]
                current_hidden_states = current_hidden_states.unsqueeze(0).expand(num_heads, -1, -1)
                cos_sim = F.cosine_similarity(attend_token_hidden_state, current_hidden_states, dim=-1)
                external_similarity.append(cos_sim)
            external_similarity = torch.cat(external_similarity, dim=1).T.cpu().float()
        else:
            external_similarity = torch.zeros(response_len, 0)

        if len(knowledge_ffns) > 0:
            knowledge_diff = torch.stack(
                [divergences[layer_idx][0, token_positions].cpu().float() for layer_idx in knowledge_ffns],
                dim=1,
            )
        else:
            knowledge_diff = torch.zeros(response_len, 0)

        return external_similarity, knowledge_diff

    def get_score(self, queries_batch: List[LLMResponse], outputs_batch: List[LLMOutputs]) -> List[List[float]]:
        features = []
        for outputs in outputs_batch:
            external_similarity, knowledge_diff = self._get_external_similarity_and_knowledge_diff_for_sample(outputs)
            features.append(knowledge_diff.flatten().tolist() + external_similarity.flatten().tolist())
        return features

    def reshape(self, features: List[List[float]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        knowledge_diff_batch = []
        external_similarity_batch = []
        k_len = len(self.knowledge_ffns)
        n_len = len(self.copy_heads)
        for sample in features:
            flat = torch.tensor(sample, dtype=torch.float)
            total_len = flat.numel()
            A = total_len // (k_len + n_len)
            assert A * (k_len + n_len) == total_len, (
                f"Feature length mismatch: got {total_len}, expected multiple of {k_len + n_len} "
                f"(layers={k_len}, heads={n_len})"
            )
            knowledge_flat = flat[: A * k_len]
            external_flat = flat[A * k_len :]
            knowledge_diff_batch.append(knowledge_flat.view(A, k_len))
            external_similarity_batch.append(external_flat.view(A, n_len))
        return knowledge_diff_batch, external_similarity_batch

    def _select_columns(self, knowledge_diff: torch.Tensor, external_similarity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_idx = self._ranked_layer_idx[: self.top_k]
        head_idx = self._ranked_head_idx[: self.top_n]
        return knowledge_diff[:, layer_idx], external_similarity[:, head_idx]

    def scale(self, knowledge_diff_batch, external_similarity_batch) -> List[List[float]]:
        results = []
        for knowledge_diff, external_similarity in zip(knowledge_diff_batch, external_similarity_batch):
            knowledge_diff, external_similarity = self._select_columns(knowledge_diff, external_similarity)
            knowledge_score = knowledge_diff.sum(dim=-1).mean()
            external_score = external_similarity.sum(dim=-1).mean()
            knowledge_score = _minmax(knowledge_score.item(), self.knowledge_diff_min_max_scale)
            external_score = _minmax(external_score.item(), self.external_similarity_min_max_scale)
            results.append((knowledge_score, -external_score))
        return results

    @staticmethod
    def sort_copy_heads_and_ffns(knowledge_diff_batch, external_similarity_batch, y):
        external_similarity, knowledge_diff = [], []
        for kd, es in zip(knowledge_diff_batch, external_similarity_batch):
            external_similarity.append(es.mean(axis=0).numpy())
            knowledge_diff.append(kd.mean(axis=0).numpy())

        external_similarity = np.stack(external_similarity, axis=0)
        knowledge_diff = np.stack(knowledge_diff, axis=0)

        ecs_roc_auc = _column_auroc(y, external_similarity, invert=True)
        sorted_heads = sorted(range(len(ecs_roc_auc)), key=lambda i: ecs_roc_auc[i], reverse=True)

        pks_roc_auc = _column_auroc(y, knowledge_diff, invert=False)
        sorted_layers = sorted(range(len(pks_roc_auc)), key=lambda i: pks_roc_auc[i], reverse=True)
        return sorted_heads, sorted_layers

    def fit_hyperparameters(self, stream: Iterable[LLMResponse], outputs_provider: LLMOutputProvider, y_val: List[int]):
        features_batch = []
        for f in self.transform(stream=stream, outputs_provider=outputs_provider):
            features_batch.extend(f)
        knowledge_diff_batch, external_similarity_batch = self.reshape(features_batch)

        sorted_heads, sorted_layers = self.sort_copy_heads_and_ffns(
            knowledge_diff_batch, external_similarity_batch, y_val
        )
        self._ranked_head_idx = list(sorted_heads)
        self._ranked_layer_idx = list(sorted_layers)
        ranked_copy_heads = [self.copy_heads[idx] for idx in sorted_heads]
        ranked_knowledge_ffns = [self.knowledge_ffns[idx] for idx in sorted_layers]
        logger.info(f"Copy heads were ordered in the following way: {ranked_copy_heads}")
        logger.info(f"Knowledge FFNs were ordered in the following way: {ranked_knowledge_ffns}")

        ranked_knowledge_diff = [kd[:, sorted_layers].numpy() for kd in knowledge_diff_batch]
        ranked_external_similarity = [es[:, sorted_heads].numpy() for es in external_similarity_batch]

        def objective(trial):
            top_n = trial.suggest_int("top_n", 1, len(self.copy_heads))
            top_k = trial.suggest_int("top_k", 1, len(self.knowledge_ffns))
            beta = trial.suggest_float("beta", 0.1, 2.0, step=0.1)
            kd_sums, es_sums, kd_bounds, es_bounds = _sum_bounds(
                ranked_knowledge_diff, ranked_external_similarity, top_k, top_n
            )
            pred = []
            for kd_sum, es_sum in zip(kd_sums, es_sums):
                kd = _minmax(float(kd_sum.mean()), kd_bounds)
                es = _minmax(float(es_sum.mean()), es_bounds)
                pred.append(self.alpha * kd + beta * (-es))
            return roc_auc_score(y_val, pred)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_params
        self.top_n = int(best_params["top_n"])
        self.top_k = int(best_params["top_k"])
        self.alpha = 1.0
        self.beta = float(best_params["beta"])
        _, _, self.knowledge_diff_min_max_scale, self.external_similarity_min_max_scale = _sum_bounds(
            ranked_knowledge_diff,
            ranked_external_similarity,
            self.top_k,
            self.top_n,
        )
        logger.info(
            f"The following hyperparameters were selected: alpha = {self.alpha}, beta = {self.beta}, "
            f"top_n = {self.top_n}, top_k = {self.top_k}"
        )

    def predict_score(self, X: List[List[float]]) -> List[float]:
        features = self.scale(*self.reshape(X))
        features = np.array(features)
        pred = self.alpha * features[:, 0] + self.beta * features[:, 1]
        return pred.tolist()

    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: List[List[float]],
        y_val: List[int],
    ) -> "WhiteBoxHalDetMethod":
        return self

    def reset(self):
        pass
