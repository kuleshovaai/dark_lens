import sys
import importlib
import numpy as np
from transformers import PreTrainedModel
from src.llm.wrappers import Wrapper
from typing import List, Tuple, Dict, Any
from functools import wraps
from .transformers_4_57_3 import eager_attention_forward


def make_eager_wrapper(selected_heads: List[Tuple[int, int]]):
    def wrapper(original_eager_fn):
        @wraps(original_eager_fn)
        def modified_eager_fn(*args, **kwargs):
            kwargs['selected_heads'] = selected_heads
            result = eager_attention_forward(*args, **kwargs)
            return result
        return modified_eager_fn
    return wrapper

class AttentionModificationWrapper(Wrapper):
    def __init__(self, llm_model: PreTrainedModel, heads: List[Tuple[int, int]]):
        self.llm_model = llm_model
        self.selected_heads = dict()
        layers = sorted(np.unique([x[0] for x in heads]).tolist())
        for layer in layers:
            self.selected_heads[layer] = sorted([x[1] for x in heads if x[0] == layer])
        self._patched_modules: Dict[str, Any] = {}
        
    def enable(self):
        """Enable the attention modification."""
        for i, layer in enumerate(self.llm_model.model.layers):
            if i not in self.selected_heads.keys():
                continue
            attn_module = layer.self_attn
            module_file = attn_module.__class__.__module__
            module_obj = sys.modules[module_file]
            key = f"{module_file}.eager_attention_forward"
            if key not in self._patched_modules:
                original_fn = getattr(module_obj, 'eager_attention_forward')
                wrapped_fn = make_eager_wrapper(self.selected_heads[i])(original_fn)
                self._patched_modules[key] = (original_fn, module_obj)
                setattr(module_obj, 'eager_attention_forward', wrapped_fn)
    
    def disable(self):
        for key, (original_fn, module_obj) in self._patched_modules.items():
            module_name, func_name = key.rsplit('.', 1)
            setattr(module_obj, func_name, original_fn)
        self._patched_modules.clear()
    
    def collect(self) -> Dict[str, Any]:
        return {}
    
    def restore(self):
        self.disable()


# class AttentionModificationWrapper(Wrapper):
#     def __init__(self, llm_model: PreTrainedModel, heads: List[Tuple[int, int]]):
#         self.llm_model = llm_model
#         self.heads = heads

#     def apply_hooks(self, heads_idxs: List[int]) -> Callable:
#         def k_hook(module, inputs, output):
#             k_scale = torch.ones_like(output[0, :, 0])
#             k_scale[0] = 1000
#             k_scale = torch.diag(k_scale)
#             output[..., heads_idxs] = torch.einsum("ij,bjk->bik", k_scale, output[..., heads_idxs])
#             return output
#         return k_hook

#     def collect(self) -> Dict[str, Any]:
#         return {}

#     def enable(self):
#         layers = defaultdict(list)
#         for layer, head in self.heads:
#             layers[layer].append(head)
#         self._handlers = []
#         for layer, heads in layers.items():
#             head_dim = self.llm_model.model.layers[layer].self_attn.head_dim
#             heads_idxs = []
#             for head in heads:
#                 heads_idxs.extend([idx for idx in range(head_dim * head, head_dim * (head + 1))])
#             k_hook = self.apply_hooks(heads_idxs)
#             k_handler = self.llm_model.model.layers[layer].self_attn.k_proj.register_forward_hook(k_hook)
#             self._handlers.append(k_handler)
    
#     def disable(self):
#         if not "_handlers" in dir(self) or not self._handlers:
#             return
#         for handler in self._handlers:
#             handler.remove()
#         self._handlers = []

#     def restore(self):
#        self.disable()
