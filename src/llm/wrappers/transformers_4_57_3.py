from typing import Optional, Tuple, Callable, List
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.deprecation import deprecate_kwarg


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[TransformersKwargs],
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    anchor_hiddens = hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    mature_hiddens = hidden_states

    return hidden_states, (anchor_hiddens, mature_hiddens)


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def gemma3_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Gemma3 double-norm decoder forward with mid/final residual for PKS."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    anchor_hiddens = hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    mature_hiddens = hidden_states
    return hidden_states, (anchor_hiddens, mature_hiddens)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    selected_heads: List[int],
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups) # type: ignore
    value_states = repeat_kv(value, module.num_key_value_groups) # type: ignore
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    else:
        causal_mask = None
        
    b_size, num_heads, q_len, k_len = attn_weights.shape
    if causal_mask is not None:
        attend_mask = (causal_mask == 0)  # (b, 1, q_len, k_len)
        first_attend = attend_mask.any(dim=2)  # (b, 1, k_len)
        positions = torch.arange(k_len, device=attn_weights.device).expand(b_size, 1, k_len)
        masked_positions = torch.where(first_attend, positions, torch.tensor(k_len, device=attn_weights.device))
        bos_pos = masked_positions.min(dim=-1)[0]  # (b, 1)
    else:
        bos_pos = torch.zeros((b_size, 1), dtype=torch.long, device=attn_weights.device)
    bos_pos = bos_pos.squeeze(-1)
    batch_idx = torch.arange(b_size, device=attn_weights.device)
    query_idx = torch.arange(q_len, device=attn_weights.device).view(1, q_len).expand(b_size, q_len)
    attn_weights[:, selected_heads, :, :] = float("-inf")
    attn_weights[batch_idx[:, None], selected_heads, query_idx, bos_pos[:, None]] = 0.0

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
