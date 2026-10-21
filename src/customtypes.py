from typing import Tuple, List, TypedDict, Union, Sequence, Optional
from dataclasses import dataclass
import torch
from transformers import BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithPast

@dataclass
class CausalLMOutputWithExtras(CausalLMOutputWithPast):
    divergences: Optional[Tuple[torch.FloatTensor]] = None
    after_layer_entropy: Optional[Tuple[torch.FloatTensor, ...]] = None
    after_layer_greedy_probs: Optional[Tuple[torch.FloatTensor, ...]] = None

LLMOutputs = Tuple[CausalLMOutputWithExtras, BatchEncoding, BatchEncoding]
Prompts = List[str]
Responses = List[str]

class LLMResponse(TypedDict):
    prompt: str
    response: List[str]

Index = Union[slice, int, Sequence[int]]
