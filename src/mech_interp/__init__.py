"""Functions to work with TransformerLens and various stats of LLM inner states."""

from .light_matter import (
    aggregate_masked,
    build_null_space_projection,
    build_random_null_space_projection,
    build_top_space_projection,
    light_matter_curve,
    light_matter_ratio,
    light_matter_ratio_masked,
    light_matter_ratio_per_layer,
    light_matter_token_curve,
    ensure_tl_unembedding_shape,
    project_out_and_renormalize,
)
from .jsd import (
    jensen_shannon_divergence,
    jsd_curve,
    jsd_token_curve,
    logit_lens_jsd_by_token,
    logit_lens_jsd_per_layer,
)
from .l2_norm import (
    l2_norm_curve,
    l2_norm_token_curve,
    l2_norm_token_curve_detailed,
    residual_l2_norm,
    residual_l2_norm_per_layer,
)
from .tokens import encode_prompt_response, char_offsets_for_token_ids, response_token_ids
from .load_model import load_tl_model
from .sink_rate import sink_rate_curve

__all__ = [
    "aggregate_masked",
    "build_null_space_projection",
    "build_random_null_space_projection",
    "build_top_space_projection",
    "char_offsets_for_token_ids",
    "light_matter_curve",
    "light_matter_ratio",
    "light_matter_ratio_masked",
    "light_matter_ratio_per_layer",
    "light_matter_token_curve",
    "encode_prompt_response",
    "ensure_tl_unembedding_shape",
    "jensen_shannon_divergence",
    "jsd_curve",
    "jsd_token_curve",
    "logit_lens_jsd_by_token",
    "l2_norm_curve",
    "l2_norm_token_curve",
    "l2_norm_token_curve_detailed",
    "load_tl_model",
    "project_out_and_renormalize",
    "logit_lens_jsd_per_layer",
    "residual_l2_norm",
    "residual_l2_norm_per_layer",
    "response_token_ids",
    "sink_rate_curve",
]
