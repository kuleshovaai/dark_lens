import types
from functools import wraps
import torch.nn.functional as F
from transformers import PreTrainedModel
from .wrapper import Wrapper
from .transformers_4_57_3 import decoder_layer_forward, gemma3_decoder_layer_forward


def calculate_dist(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)  
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)  

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)

    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)  
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1) 

    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)

    js_divs = 0.5 * (kl1 + kl2)  

    return js_divs * 1e6


def _resolve_decoder_stack(llm_model: PreTrainedModel):
    """Return (layers, final_norm) for Llama-style and Gemma3 multimodal stacks."""
    root = llm_model.model
    if hasattr(root, "layers"):
        return root.layers, root.norm
    language_model = getattr(root, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model.layers, language_model.norm
    raise AttributeError(
        f"Could not locate decoder layers on {type(llm_model).__name__} "
        f"(looked for model.layers and model.language_model.layers)."
    )


class WrapperForResidualStreamLogits(Wrapper):
    def __init__(self, llm_model: PreTrainedModel):
        self.dists = []
        self._enabled = False
        self.llm_model = llm_model
        self._layers, self._final_norm = _resolve_decoder_stack(llm_model)
        for layer in self._layers:
            self._wrap_layer(layer)

    def collect(self):
        dists = self.dists
        self.dists = []
        return {"divergences": tuple(dists)}

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def restore(self):
        for layer in self._layers:
            if hasattr(layer, "_original_forward"):
                layer.forward = layer._original_forward
                del layer._original_forward

    def _wrap_layer(self, layer):
        orig_forward = layer.forward
        layer_forward = (
            gemma3_decoder_layer_forward
            if hasattr(layer, "pre_feedforward_layernorm")
            else decoder_layer_forward
        )

        @wraps(orig_forward)
        def wrapped_forward(this_layer, *args, **kwargs):
            if not self._enabled:
                return orig_forward(*args, **kwargs)
            output, (anchor_h, mature_h) = layer_forward(this_layer, *args, **kwargs)
            anchor_h = self._norm(anchor_h)
            mature_h = self._norm(mature_h)
            anchor_logits = self._compute_logits(anchor_h)
            mature_logits = self._compute_logits(mature_h)
            dist = calculate_dist(
                mature_logits,
                anchor_logits,
            )
            self.dists.append(dist)
            return output
        layer._original_forward = orig_forward
        layer.forward = types.MethodType(wrapped_forward, layer)
    
    def _compute_logits(self, hidden_states, logits_to_keep=0):
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.llm_model.lm_head(hidden_states[:, slice_indices, :])
        return logits
        
    def _norm(self, hiddens):
        return self._final_norm(hiddens)
