from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from src.customtypes import LLMOutputs, LLMResponse
from src.hallucination_detection.hallucination_detection import WhiteBoxHalDetMethod
from src.hallucination_detection.redeep.redeep import _column_auroc, _minmax
from src.llm import LLMConfig, LLMOutputProvider
from src.mech_interp.light_matter import unembedding_null_basis, unembedding_right_basis

DEFAULT_NULL_DIMS = (8, 16, 32, 64, 128)
BETA_GRID = tuple(round(x, 1) for x in np.arange(0.1, 2.01, 0.1))


def _minmax_array(values: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _auroc_rows(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Binary AUROC for each row of ``scores`` with shape ``[n_configs, n_samples]``."""
    y = np.asarray(labels, dtype=bool)
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[0], 0.5, dtype=np.float64)
    order = np.argsort(scores, axis=1, kind="mergesort")
    y_sorted = y[order]
    n_samples = scores.shape[1]
    ranks = np.arange(1, n_samples + 1, dtype=np.float64)
    sum_pos = (y_sorted * ranks).sum(axis=1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass
class DarkLens(WhiteBoxHalDetMethod):
    """DarkLens: token-mean PKS plus a light-matter residual score.

    Token-mean PKS is taken over ranked FFN layers. The light-matter term is
    the residual ratio ``1 - ||P_null h|| / ||h||`` (higher = more
    hallucination, matching ``NullSpaceProjection``). Both terms use
    prediction-aligned residual positions (the state that generates each
    answer token). ``components`` selects the ablation:

        pks   — minmax(PKS), ``top_k`` chosen on validation
        lm    — minmax(LM), layer and null-space rank chosen on validation
        both  — H = α · minmax(PKS) + β · minmax(LM)

    For ``both``, the LM layer, null-space rank, PKS ``top_k``, and ``beta``
    are selected jointly on validation. Each setting is fit on its own score,
    so a gain of ``both`` over ``pks`` and over ``lm`` is evidence that both
    terms contribute. Pass a singleton ``lm_layers`` / ``null_dims`` to hold
    those axes fixed (layer / null-rank ablation sweeps).

    ``spectrum`` chooses the rank-``null_dim`` subspace used by the
    light-matter term. ``dark`` is the trailing singular band. ``middle`` is
    the same-rank band centered in the spectrum. ``random`` is a fixed
    orthonormal basis of that rank, drawn once from ``basis_seed``; smaller
    candidate ranks are leading slices of the largest draw.
    """

    llm_config: LLMConfig
    unembedding_matrix: torch.Tensor = field(repr=False, compare=False)

    batch_size: int = 1
    use_cache: bool = True
    cache_dir: str = "cache/dark_lens"

    knowledge_ffns: Optional[List[int]] = None
    spectrum: Literal["dark", "middle", "random"] = "dark"
    basis_seed: int = 0
    null_dims: List[int] = field(default_factory=lambda: list(DEFAULT_NULL_DIMS))
    aggregation: Literal["max", "min", "mean", "last"] = "mean"
    last_k_layers: Optional[int] = None
    # If set, validation only searches these LM residual layers (else last_k_layers / all).
    lm_layers: Optional[List[int]] = None

    components: Literal["pks", "lm", "both"] = "both"
    top_k: Optional[int] = None
    alpha: Optional[float] = 1.0
    beta: Optional[float] = 0.2
    layer: Optional[int] = None
    null_dim: Optional[int] = None

    knowledge_diff_min_max_scale: Tuple[float, float] = (0.0, 1.0)
    light_matter_min_max_scale: Tuple[float, float] = (0.0, 1.0)

    need_fit: bool = True

    _ranked_layer_idx: List[int] = field(init=False, repr=False)
    _v_null: Optional[torch.Tensor] = field(init=False, repr=False, compare=False)
    _vh: Optional[torch.Tensor] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        super().__init__(batch_size=self.batch_size, use_cache=self.use_cache, cache_dir=self.cache_dir)
        if self.unembedding_matrix is None:
            raise ValueError("unembedding_matrix must be provided during initialization.")

        n_layers = self.llm_config.num_hidden_layers
        if self.knowledge_ffns is None:
            self.knowledge_ffns = list(range(n_layers))
        else:
            self.knowledge_ffns = sorted(int(layer) for layer in self.knowledge_ffns)
        self.null_dims = sorted({int(dim) for dim in self.null_dims})
        if not self.null_dims:
            raise ValueError("null_dims must contain at least one positive rank.")
        if any(dim <= 0 for dim in self.null_dims):
            raise ValueError(f"null_dims must be positive, got {self.null_dims}")

        if self.components not in ("pks", "lm", "both"):
            raise ValueError(f"components must be one of pks, lm, both; got {self.components!r}")
        if self.spectrum not in ("dark", "middle", "random"):
            raise ValueError(f"spectrum must be dark, middle, or random, got {self.spectrum!r}")

        if self.top_k is None:
            self.top_k = len(self.knowledge_ffns)
        if self.alpha is None:
            self.alpha = 1.0
        if self.beta is None:
            self.beta = 0.2
        if self.null_dim is None:
            self.null_dim = 32 if 32 in self.null_dims else self.null_dims[len(self.null_dims) // 2]
        if self.null_dim not in self.null_dims:
            raise ValueError(f"null_dim={self.null_dim} is not in candidate null_dims={self.null_dims}")
        if self.lm_layers is not None:
            self.lm_layers = sorted({int(layer) for layer in self.lm_layers})
            if not self.lm_layers:
                raise ValueError("lm_layers must contain at least one layer index.")
            bad = [i for i in self.lm_layers if i < 0 or i >= n_layers]
            if bad:
                raise ValueError(
                    f"lm_layers {bad} outside 0..{n_layers - 1} for this model"
                )
        if self.layer is None:
            self.layer = (
                self.lm_layers[-1] if self.lm_layers is not None else n_layers - 1
            )
        if self.lm_layers is not None and self.layer not in self.lm_layers:
            raise ValueError(
                f"layer={self.layer} is not in candidate lm_layers={self.lm_layers}"
            )

        self._ranked_layer_idx = list(range(len(self.knowledge_ffns)))
        self._prepare_null_spaces()

    def _prepare_null_spaces(self) -> None:
        # Dark: one thin basis for the largest rank. Smaller null_dims are trailing
        # slices, since ||P_k h|| = ||V_k^T h|| for V_k = V_max[:, -k:].
        # Middle: the full descending basis, so each rank can take its own centered band.
        # Random: one QR basis of the largest rank. Smaller ranks are leading slices.
        self._vh = None
        max_dim = max(self.null_dims)
        if self.spectrum == "dark":
            self._v_null = unembedding_null_basis(self.unembedding_matrix, null_dim=max_dim).contiguous()
            return
        if self.spectrum == "middle":
            self._v_null = None
            self._vh = unembedding_right_basis(self.unembedding_matrix).contiguous()
            return
        weight = self.unembedding_matrix
        d_model = weight.shape[1] if weight.shape[0] > weight.shape[1] else weight.shape[0]
        if max_dim > d_model:
            raise ValueError(f"null_dim={max_dim} exceeds d_model={d_model}")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.basis_seed))
        gauss = torch.randn(d_model, max_dim, generator=generator)
        q, _ = torch.linalg.qr(gauss, mode="reduced")
        self._v_null = q.contiguous()

    def _lm_layer_indices(self, n_layers: Optional[int] = None) -> List[int]:
        """LM layers available for validation selection.

        Prefer an explicit ``lm_layers`` list when set. Otherwise all transformer
        layers are considered, or the final ``last_k_layers`` if that is set.
        """
        n_layers = n_layers or self.llm_config.num_hidden_layers
        if self.lm_layers is not None:
            return list(self.lm_layers)
        if self.last_k_layers is None:
            return list(range(n_layers))
        if self.last_k_layers <= 0:
            raise ValueError(f"last_k_layers must be positive, got {self.last_k_layers}")
        if self.last_k_layers > n_layers:
            raise ValueError(
                f"last_k_layers={self.last_k_layers} exceeds model layer count {n_layers}"
            )
        return list(range(n_layers - self.last_k_layers, n_layers))

    def _generation_positions(self, prompt_len: int, response_len: int, device) -> torch.Tensor:
        start = prompt_len - 1
        return torch.arange(start, start + response_len, device=device)

    def _pks_mean_for_sample(self, outputs: LLMOutputs) -> torch.Tensor:
        llm_forward_output, prompt_ids, answer_ids = outputs
        divergences = llm_forward_output.divergences
        if divergences is None:
            raise ValueError("Expected to get `divergences`, but got None.")
        response_len = int(answer_ids.attention_mask.sum().item())
        n_ffns = len(self.knowledge_ffns)
        if response_len == 0 or n_ffns == 0:
            return torch.zeros(n_ffns)
        prompt_len = prompt_ids.input_ids.shape[1]
        token_positions = self._generation_positions(prompt_len, response_len, divergences[0].device)
        knowledge_diff = torch.stack(
            [divergences[layer_idx][0, token_positions].float() for layer_idx in self.knowledge_ffns],
            dim=0,
        )
        return knowledge_diff.mean(dim=1).cpu()

    def _aggregate_token_scores(self, ratio: torch.Tensor) -> torch.Tensor:
        """Aggregate ``[n_layers, n_tokens]`` light-matter ratios over the response."""
        if self.aggregation == "mean":
            return ratio.mean(dim=-1)
        if self.aggregation == "min":
            return ratio.min(dim=-1).values
        if self.aggregation == "max":
            return ratio.max(dim=-1).values
        if self.aggregation == "last":
            return ratio[:, -1]
        raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

    def _lm_grid_for_sample(self, outputs: LLMOutputs) -> torch.Tensor:
        llm_forward_outputs, prompt_ids, answer_ids = outputs
        if llm_forward_outputs.hidden_states is None:
            raise ValueError("Expected to get `hidden states`, but got None.")

        n_layers = self.llm_config.num_hidden_layers
        hidden_states = llm_forward_outputs.hidden_states
        if len(hidden_states) < n_layers + 1:
            raise ValueError(
                f"Expected at least {n_layers + 1} hidden-state tensors "
                f"(embedding + transformer layers), got {len(hidden_states)}."
            )

        response_len = int(answer_ids.attention_mask.sum().item())
        if response_len == 0:
            return torch.zeros(len(self.null_dims), n_layers)
        prompt_len = prompt_ids.input_ids.shape[1]
        if prompt_len < 1:
            raise ValueError(
                "prediction-aligned LM needs at least one prompt token "
                "(last prompt position predicts the first answer token)"
            )
        token_positions = self._generation_positions(
            prompt_len, response_len, hidden_states[-1].device
        )
        h = torch.stack(
            [
                hidden_states[layer_idx + 1][0, token_positions].float()
                for layer_idx in range(n_layers)
            ],
            dim=0,
        )

        h_norm = torch.linalg.norm(h, dim=-1).clamp_min(1e-8)
        scores = []
        if self.spectrum == "middle":
            vh = self._vh.to(device=h.device, dtype=h.dtype)
            n_sv = vh.shape[0]
            for dim in self.null_dims:
                start = n_sv // 2
                basis = vh[start : start + dim].T
                null_norm = torch.linalg.norm(h @ basis, dim=-1)
                scores.append(self._aggregate_token_scores(1.0 - null_norm / h_norm))
        else:
            # Dark uses the trailing columns. Random uses the leading columns.
            v_basis = self._v_null.to(device=h.device, dtype=h.dtype)
            projected = h @ v_basis
            take_trailing = self.spectrum == "dark"
            for dim in self.null_dims:
                coords = projected[..., -dim:] if take_trailing else projected[..., :dim]
                null_norm = torch.linalg.norm(coords, dim=-1)
                scores.append(self._aggregate_token_scores(1.0 - null_norm / h_norm))
        return torch.stack(scores, dim=0).cpu()

    @property
    def config(self):
        # Omit val-selected hyperparameters so the transform cache is shared across folds.
        return {
            "llm_model": repr(self.llm_config),
            "llm_model_config": self.llm_config.text_config_to_dict(),
            "knowledge_ffns": self.knowledge_ffns,
            "null_dims": self.null_dims,
            "aggregation": self.aggregation,
            # Bust caches computed with token-presence LM positions.
            "lm_positions": "prediction_aligned",
            # ``components`` only changes which cached features are scored.
            # ``spectrum`` changes the light-matter features, so only a non-default
            # band is part of the cache key. Dark runs keep the previous key.
            **(
                {"spectrum": self.spectrum, **({"basis_seed": int(self.basis_seed)} if self.spectrum == "random" else {})}
                if self.spectrum != "dark"
                else {}
            ),
        }

    def get_score(self, queries_batch: List[LLMResponse], outputs_batch: List[LLMOutputs]) -> List[List[float]]:
        features = []
        for outputs in outputs_batch:
            pks = self._pks_mean_for_sample(outputs)
            lm = self._lm_grid_for_sample(outputs)
            features.append(pks.flatten().tolist() + lm.flatten().tolist())
        return features

    def reshape(self, features: List[List[float]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        knowledge_diff_batch = []
        light_matter_batch = []
        k_len = len(self.knowledge_ffns)
        n_dims = len(self.null_dims)
        n_layers = self.llm_config.num_hidden_layers
        expected = k_len + n_dims * n_layers
        for sample in features:
            flat = torch.tensor(sample, dtype=torch.float)
            if flat.numel() != expected:
                raise ValueError(
                    f"Feature length mismatch: got {flat.numel()}, expected {expected} "
                    f"(pks={k_len}, null_dims={n_dims}, layers={n_layers})"
                )
            knowledge_diff_batch.append(flat[:k_len])
            light_matter_batch.append(flat[k_len:].view(n_dims, n_layers))
        return knowledge_diff_batch, light_matter_batch

    def _null_dim_index(self) -> int:
        try:
            return self.null_dims.index(int(self.null_dim))
        except ValueError as exc:
            raise ValueError(
                f"Selected null_dim={self.null_dim} is not in candidate null_dims={self.null_dims}"
            ) from exc

    def scale(
        self,
        knowledge_diff_batch: List[torch.Tensor],
        light_matter_batch: List[torch.Tensor],
    ) -> List[List[float]]:
        layer_idx = self._ranked_layer_idx[: self.top_k]
        dim_idx = self._null_dim_index()
        lm_layer = int(self.layer)
        results = []
        for knowledge_diff, light_matter in zip(knowledge_diff_batch, light_matter_batch):
            knowledge_score = knowledge_diff[layer_idx].sum()
            lm_score = light_matter[dim_idx, lm_layer]
            knowledge_score = _minmax(knowledge_score.item(), self.knowledge_diff_min_max_scale)
            lm_score = _minmax(lm_score.item(), self.light_matter_min_max_scale)
            results.append((knowledge_score, lm_score))
        return results

    def fit_hyperparameters(
        self,
        stream: Iterable[LLMResponse],
        outputs_provider: LLMOutputProvider,
        y_val: List[int],
    ):
        features_batch = []
        for features in self.transform(stream=stream, outputs_provider=outputs_provider):
            features_batch.extend(features)
        knowledge_diff_batch, light_matter_batch = self.reshape(features_batch)
        if len(knowledge_diff_batch) != len(y_val):
            raise ValueError(
                f"Validation feature count ({len(knowledge_diff_batch)}) does not match "
                f"label count ({len(y_val)})."
            )

        pks = np.stack([kd.numpy() for kd in knowledge_diff_batch], axis=0)
        lm = np.stack([scores.numpy() for scores in light_matter_batch], axis=0)
        pks_aucs = _column_auroc(y_val, pks, invert=False)
        sorted_layers = sorted(range(len(pks_aucs)), key=lambda i: pks_aucs[i], reverse=True)
        self._ranked_layer_idx = list(sorted_layers)
        ranked_knowledge_ffns = [self.knowledge_ffns[idx] for idx in sorted_layers]
        logger.info(f"Knowledge FFNs were ordered in the following way: {ranked_knowledge_ffns}")

        ranked_pks = pks[:, sorted_layers]
        n_pks = ranked_pks.shape[1]
        n_layers = lm.shape[-1]
        layer_indices = self._lm_layer_indices(n_layers)
        labels = np.asarray(y_val)
        alpha = 1.0
        betas = np.asarray(BETA_GRID, dtype=np.float64)

        lm_rows = []
        combo_meta = []
        for dim_idx, null_dim in enumerate(self.null_dims):
            for lm_layer in layer_indices:
                vals = lm[:, dim_idx, lm_layer]
                lm_bounds = (float(vals.min()), float(vals.max()))
                lm_rows.append(_minmax_array(vals, lm_bounds))
                combo_meta.append((int(null_dim), int(lm_layer), lm_bounds))
        lm_n = np.stack(lm_rows, axis=0)
        n_beta = betas.shape[0]
        n_combo = lm_n.shape[0]

        if self.components == "pks":
            best_auroc = -1.0
            best_top_k = n_pks
            best_bounds = (0.0, 1.0)
            for top_k in range(1, n_pks + 1):
                pks_sum = ranked_pks[:, :top_k].sum(axis=1)
                pks_bounds = (float(pks_sum.min()), float(pks_sum.max()))
                pks_n = _minmax_array(pks_sum, pks_bounds)
                auroc = float(_auroc_rows(labels, pks_n[None, :])[0])
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_top_k = top_k
                    best_bounds = pks_bounds
            self.top_k = int(best_top_k)
            self.alpha = 1.0
            self.beta = 0.0
            self.knowledge_diff_min_max_scale = best_bounds
            logger.info(
                f"PKS-only hyperparameters: top_k = {self.top_k} "
                f"(validation AUROC {best_auroc:.4f})"
            )
            return

        if self.components == "lm":
            aucs = _auroc_rows(labels, lm_n)
            combo_idx = int(np.argmax(aucs))
            best_auroc = float(aucs[combo_idx])
            null_dim, lm_layer, lm_bounds = combo_meta[combo_idx]
            self.alpha = 0.0
            self.beta = 1.0
            self.layer = int(lm_layer)
            self.null_dim = int(null_dim)
            self.light_matter_min_max_scale = lm_bounds
            logger.info(
                f"LM-only hyperparameters: lm_layer = {self.layer}, null_dim = {self.null_dim} "
                f"(validation AUROC {best_auroc:.4f})"
            )
            return

        best_auroc = -1.0
        best = {
            "top_k": n_pks,
            "beta": 0.2,
            "layer": layer_indices[-1],
            "null_dim": self.null_dim,
            "pks_bounds": (0.0, 1.0),
            "lm_bounds": (0.0, 1.0),
        }
        for top_k in range(1, n_pks + 1):
            pks_sum = ranked_pks[:, :top_k].sum(axis=1)
            pks_bounds = (float(pks_sum.min()), float(pks_sum.max()))
            pks_n = _minmax_array(pks_sum, pks_bounds)
            pred = pks_n[None, None, :] + betas[:, None, None] * lm_n[None, :, :]
            aucs = _auroc_rows(labels, pred.reshape(n_beta * n_combo, -1)).reshape(n_beta, n_combo)
            beta_idx, combo_idx = np.unravel_index(int(aucs.argmax()), aucs.shape)
            auroc = float(aucs[beta_idx, combo_idx])
            if auroc > best_auroc:
                null_dim, lm_layer, lm_bounds = combo_meta[int(combo_idx)]
                best_auroc = auroc
                best = {
                    "top_k": top_k,
                    "beta": float(betas[int(beta_idx)]),
                    "layer": lm_layer,
                    "null_dim": null_dim,
                    "pks_bounds": pks_bounds,
                    "lm_bounds": lm_bounds,
                }

        self.top_k = int(best["top_k"])
        self.alpha = alpha
        self.beta = float(best["beta"])
        self.layer = int(best["layer"])
        self.null_dim = int(best["null_dim"])
        self.knowledge_diff_min_max_scale = best["pks_bounds"]
        self.light_matter_min_max_scale = best["lm_bounds"]
        logger.info(
            f"The following hyperparameters were selected: alpha = {self.alpha}, beta = {self.beta}, "
            f"top_k = {self.top_k}, lm_layer = {self.layer}, null_dim = {self.null_dim} "
            f"(validation AUROC {best_auroc:.4f})"
        )

    def predict_score(self, X: List[List[float]]) -> List[float]:
        features = self.scale(*self.reshape(X))
        features = np.array(features)
        if self.components == "pks":
            pred = features[:, 0]
        elif self.components == "lm":
            pred = features[:, 1]
        else:
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
        # Keep val-selected hyperparameters; evaluate() calls reset() after transform.
        pass
