import types
from functools import wraps

import torch.nn.functional as F
from transformers import PreTrainedModel

from .redeep_wrapper import _resolve_decoder_stack
from .transformers_4_57_3 import decoder_layer_forward, gemma3_decoder_layer_forward
from .wrapper import Wrapper


class WrapperForAfterLayerLogits(Wrapper):
    """Per-layer entropy and greedy probability of the residual stream after each block."""

    def __init__(self, llm_model: PreTrainedModel):
        self.entropies = []
        self.greedy_probs = []
        self._enabled = False
        self.llm_model = llm_model
        self._layers, self._final_norm = _resolve_decoder_stack(llm_model)
        for layer in self._layers:
            self._wrap_layer(layer)

    def collect(self):
        entropies = tuple(self.entropies)
        greedy_probs = tuple(self.greedy_probs)
        self.entropies = []
        self.greedy_probs = []
        return {
            "after_layer_entropy": entropies,
            "after_layer_greedy_probs": greedy_probs,
        }

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
            output, (_, mature_h) = layer_forward(this_layer, *args, **kwargs)
            logits = self._compute_logits(self._norm(mature_h))
            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            greedy = probs.max(dim=-1).values
            self.entropies.append(entropy)
            self.greedy_probs.append(greedy)
            return output

        layer._original_forward = orig_forward
        layer.forward = types.MethodType(wrapped_forward, layer)

    def _compute_logits(self, hidden_states):
        return self.llm_model.lm_head(hidden_states)

    def _norm(self, hiddens):
        return self._final_norm(hiddens)
