from typing import List, Sequence, Tuple, Literal, Dict
import itertools
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import BatchEncoding

from src.customtypes import Index, Prompts, Responses, LLMOutputs, CausalLMOutputWithExtras

def to_py_obj(obj):
    if isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        raise NotImplemented

def pad(
    encoded_inputs: Dict[str, List],
    max_length: int,
    pad_token_id: int = 0,
    padding_side: Literal["left", "right"] = "right",
) -> dict:
    difference = max_length - len(encoded_inputs["input_ids"])
    padding_side = padding_side if padding_side is not None else self.padding_side

    if padding_side == "right":
        encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [pad_token_id] * difference
        if "attention_mask" in encoded_inputs.keys():
            encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
        if "offset_mapping" in encoded_inputs.keys():
            encoded_inputs["offset_mapping"] = encoded_inputs["offset_mapping"] + [[0, 0]] * difference
        if "special_tokens_mask" in encoded_inputs.keys():
            encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
    elif padding_side == "left":
        encoded_inputs["input_ids"] = [pad_token_id] * difference + encoded_inputs["input_ids"]
        if "attention_mask" in encoded_inputs.keys():
            encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
        if "offset_mapping" in encoded_inputs.keys():
            encoded_inputs["offset_mapping"] = [[0, 0]] * difference + encoded_inputs["offset_mapping"]
        if "special_tokens_mask" in encoded_inputs.keys():
            encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
    else:
        raise ValueError(f"Invalid padding strategy:{padding_side}")
    return encoded_inputs

# minimized implementation from transformers.tokenization_utils_base
def concat_batch_encodings(
    encoded_inputs: tuple[BatchEncoding, ...], 
    pad_token_id: int = 0,
    padding_side: Literal["left", "right"] = "right",
) -> Tuple[BatchEncoding, List[List[int]]]:
    # If we have a list of dicts, let's convert it in a dict of lists
    # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
    if isinstance(encoded_inputs, (list, tuple)):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0]}

    # If we have PyTorch tensors as inputs, we cast them as python objects
    # and rebuild them afterwards if no return_tensors is specified
    # Note that we lose the specific device the tensor may be on for PyTorch
    for key, value in encoded_inputs.items():
        encoded_inputs[key] = to_py_obj(value)

    length_per_block = [len(block[0]) for block in encoded_inputs["input_ids"]]
    flattened_inputs = {k: [*itertools.chain(*v)] for k, v in encoded_inputs.items()}
    max_length = max([len(input_ids) for input_ids in flattened_inputs["input_ids"]])
    num_samples = len(flattened_inputs["input_ids"])
    batch_outputs = {}
    for i in range(num_samples):
        inputs = {k: v[i] for k, v in flattened_inputs.items()}
        outputs = pad(
            inputs,
            max_length=max_length,
            pad_token_id=pad_token_id,
            padding_side=padding_side,
        )
        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)

    padding_configs = []
    for length in length_per_block:
        pad_amt = max_length - length
        pad_config = [0, 0, 0, 0]
        if padding_side == "right":
            pad_config[1] = pad_amt
        else:
            pad_config[0] = pad_amt
        padding_configs.append(pad_config)

    return BatchEncoding(batch_outputs, tensor_type="pt"), padding_configs

def concat_outputs(
    *outputs: LLMOutputs, 
    pad_token_id: int,
) -> LLMOutputs:
    """
    Merges multiple LLM output tuples, padding logits and encodings as necessary.
    """
    if not outputs:
        raise ValueError("At least one LLMOutputs must be provided")

    if len(outputs) == 1:
        return outputs[0]

    outs, encs, gens = zip(*outputs)

    merged_encs, padding_configs_left = concat_batch_encodings(encs, pad_token_id=pad_token_id, padding_side="left")
    merged_gens, padding_configs_right = concat_batch_encodings(gens, pad_token_id=pad_token_id, padding_side="right")

    def concat_outputs(
        tensors: List[torch.Tensor],
        padding_value: float = 0,
        dim: Tuple[int, ...] = (1, ),
    ) -> torch.Tensor:
        if not tensors:
            return torch.tensor([])
        padded_tensors = []
        for t, padding_config_left, padding_config_right in zip(
            tensors,
            padding_configs_left,
            padding_configs_right,
        ):
            pad_config = [0] * (t.ndim * 2)
            for d in dim:
                left_idx = (t.ndim - 1 - d) * 2
                right_idx = (t.ndim - 1 - d) * 2 + 1 
                pad_config[left_idx] = padding_config_left[0]
                pad_config[right_idx] = padding_config_right[1]
            padded_tensors.append(F.pad(t, pad_config, value=padding_value))
        return torch.cat(padded_tensors, dim=0)

    data = {}
    first_out = outs[0]
    for field in first_out.__dict__.keys():
        v0 = getattr(first_out, field)
        if v0 is None:
            data[field] = None
            continue
        values = [getattr(o, field) for o in outs]

        if field == "logits":
            # logits: [B, Seq, Vocab] -> pad on dim=1 (Seq)
            data[field] = concat_outputs(values, dim=(1, ))
        elif field == "past_key_values":
            data[field] = None
        elif field == "hidden_states":
            if isinstance(values[0], tuple):
                data[field] = tuple(concat_outputs([v[i] for v in values], dim=(1, )) for i in range(len(values[0])))
            else:
                data[field] = concat_outputs(values, dim=(1, ))
        elif field == "attentions":
            if isinstance(values[0], tuple):
                padded_layers = []
                for i in range(len(values[0])):
                    layer_tensors = [v[i] for v in values]
                    layer_padded = concat_outputs(layer_tensors, dim=(2, 3))
                    padded_layers.append(layer_padded)
                data[field] = tuple(padded_layers)
            else:
                att = concat_outputs(values, dim=(2, 3))
                data[field] = att
        elif isinstance(values[0], torch.Tensor):
            try:
                data[field] = concat_outputs(values, dim=(1, ))
            except:
                raise TypeError(f"Failed to concatenate values with type `torch.Tensor` of unrecognized key-word argument {field}.")
        elif isinstance(values[0], tuple):
            try:
                data[field] = tuple(concat_outputs([v[i] for v in values], dim=(1, )) for i in range(len(values[0])))
            except:
                raise TypeError(f"Failed to concatenate values with type `tuple` of unrecognized key-word argument {field}.")
        else:
            raise TypeError(f"Unsupported field '{field}' ({type(v0)})")

    merged_out = CausalLMOutputWithExtras(**data)
    return (merged_out, merged_encs, merged_gens)

def slice_outputs(
    out: CausalLMOutputWithExtras,
    idx: "Index",
) -> CausalLMOutputWithExtras:
    data = {}
    length = out.logits.size(0) # type: ignore
    if isinstance(idx, int):
        idx = torch.tensor([idx], device=out.logits.device) # type: ignore
    elif isinstance(idx, slice):
        start, stop, step = idx.indices(length)
        idx = torch.arange(start, stop, step, device=out.logits.device) # type: ignore
    elif isinstance(idx, Sequence):
        idx = torch.tensor(list(idx), device=out.logits.device) # type: ignore
    else:
        raise TypeError(f"Unsupported index type: {type(idx)}")
    for field, value in out.__dict__.items():
        if value is None:
            data[field] = None
        elif field == "logits":
            data[field] = value[idx]
        elif field == "past_key_values":
            data[field] = None
        elif field in ["hidden_states", "attentions"]:
            data[field] = tuple(v[idx] for v in value)
        elif isinstance(value, torch.Tensor):
            try:
                data[field] = value[idx]
            except:
                raise TypeError(f"Failed to slice value with type `torch.Tensor` of unrecognized key-word argument {field}.")
        elif isinstance(value, tuple):
            try:
               data[field] = tuple(v[idx] for v in value)
            except:
                raise TypeError(f"Failed to slice value with type `tuple` of unrecognized key-word argument {field}.")
        else:
            raise TypeError(
                f"Unsupported field '{field}' in CausalLMOutputWithExtras "
                f"({type(value)})"
            )
    return CausalLMOutputWithExtras(**data)

def slice_batch_encoding(
    enc: BatchEncoding,
    idx: "Index",
) -> BatchEncoding:
    data = {}   

    length = enc.input_ids.size(0)
    if isinstance(idx, int):
        idx = torch.tensor([idx]) # type: ignore
    elif isinstance(idx, slice):
        start, stop, step = idx.indices(length)
        idx = torch.arange(start, stop, step) # type: ignore
    elif isinstance(idx, Sequence):
        idx = torch.tensor(list(idx)) # type: ignore
    else:
        raise TypeError(f"Unsupported index type: {type(idx)}")
    
    for field, value in enc.items():
        if value is None:
            data[field] = None
        elif field in ["input_ids", "attention_mask", "offset_mapping", "special_tokens_mask"]:
            data[field] = value[idx]
        else:
            raise TypeError(
                f"Unsupported field '{field}' in BatchEncoding "
                f"({type(value)})"
            )
    return BatchEncoding(data)

def slice_list(lst: List, idx: "Index") -> List:
    if isinstance(idx, slice):
        return lst[idx]
    if isinstance(idx, int):
        return [lst[idx]]
    return [lst[i] for i in idx]


@dataclass
class LLMOutputsWithQuery:
    sample_ids: List[int]
    num_seqs: List[int]
    prompts: Prompts
    responses: Responses
    prompt_tokens: BatchEncoding
    response_tokens: BatchEncoding
    outputs: CausalLMOutputWithExtras

    def __getitem__(self, idx: Index) -> "LLMOutputsWithQuery":
        if isinstance(idx, slice) and (idx.start == 0 and idx.stop >= len(self)):
            return self
        return LLMOutputsWithQuery(
            sample_ids=slice_list(self.sample_ids, idx),
            num_seqs=slice_list(self.num_seqs, idx),
            prompts=slice_list(self.prompts, idx),
            responses=slice_list(self.responses, idx),
            prompt_tokens=slice_batch_encoding(self.prompt_tokens, idx),
            response_tokens=slice_batch_encoding(self.response_tokens, idx),
            outputs=slice_outputs(self.outputs, idx),
        )

    def __len__(self):
        return len(self.sample_ids)
    
    
if __name__ == "__main__":
    from src.llm import LLM
    
    llm_model = LLM(model_name="Qwen3-8B")
    text1 = "Hello, world!"
    text2 = "Hi!"
    
    tokens1 = llm_model.tokenizer(text1, return_offsets_mapping=True, return_tensors="pt")
    tokens2 = llm_model.tokenizer(text2, return_offsets_mapping=True, return_tensors="pt")
    
    print(tokens1)
    print(tokens2)
    print(concat_batch_encodings((tokens1, tokens2)))
    print(next(llm_model.generate_llm_outputs([text1], [text2])))