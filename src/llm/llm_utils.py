
import os
from typing import List, Union, Optional, Any, Dict, Callable
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


def get_model_id(model_name):
    if "Llama" in model_name:
        model_id = f"meta-llama/{model_name}"
    elif "Mistral" in model_name:
        model_id = f"mistralai/{model_name}"
    elif "Qwen" in model_name:
        model_id = f"Qwen/{model_name}"
    elif "gpt" in model_name:
        model_id = f"openai-community/{model_name}"
    elif "gemma" in model_name:
        model_id = f"google/{model_name}"
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model_id

def instantiate_model(model_name, dtype, device, attn_implementation):
    logger.info(f"Instantiating {model_name} model.")
    token = os.getenv("HUGGING_FACE_API_KEY")
    login(token=token)
    model_id = get_model_id(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        attn_implementation=attn_implementation,
    )
    logger.info(f"{model_name} model has been successfully instantiated.")
    return llm

def instantiate_tokenizer(model_name):
    logger.info(f"Instantiating {model_name} tokenizer.")
    token = os.getenv("HUGGING_FACE_API_KEY")
    login(token=token)
    model_id = get_model_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"{model_name} tokenizer has been successfully instantiated.")
    return tokenizer

def check_prompt_response_compatibility(
    user_prompt: Union[None, str, List[str], List[List[str]]],
    assistant_response: Union[None, str, List[str], List[List[str]]],
) -> None:
    def _type(x):
        if x is None:
            return None
        if isinstance(x, str):
            return "str"
        if isinstance(x, list):
            if not x:
                return "List[str]"  # empty list → treat as 1D
            if all(isinstance(i, str) for i in x):
                return "List[str]"
            if all(isinstance(i, list) and all(isinstance(j, str) for j in i) for i in x):
                return "List[List[str]]"
        raise TypeError(f"Unsupported type: {type(x)}")
    user_shape = _type(user_prompt)
    assistant_shape = _type(assistant_response)
    if user_shape is None or assistant_shape is None:
        return
    if user_shape != assistant_shape:
        raise TypeError(
            "Incompatible shapes: "
            f"user_prompt is {user_shape}, "
            f"assistant_response is {assistant_shape}."
        )

def get_conversation_dict(
    user_prompt: Optional[Union[str, List[str]]] = None,
    assistant_response: Optional[Union[str, List[str]]] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build a conversation in chat format.
    The function supports both single-turn and multi-turn conversations.

    Parameters
    ----------
    user_prompt : str or List[str]
        User message(s). If a list is provided, each element corresponds
        to one user turn in the conversation.
    assistant_response : str or List[str] or None, optional
        Assistant message(s). If a list is provided, its length must be
        equal to the number of user turns or exactly one less.
    system_prompt : str or None, optional
        Optional system instruction prepended to the conversation.

    Returns
    -------
    List[Dict[str, Any]]
        A list of messages in the format:
        {"role": "system" | "user" | "assistant", "content": str}
    """
    check_prompt_response_compatibility(user_prompt, assistant_response)
    conversation: List[Dict[str, Any]] = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})
    if user_prompt is None:
        if assistant_response is not None:
            raise TypeError(
                "When `user_prompt` is None, `assistant_response` must be None. "
                f"Got assistant_response of type {type(assistant_response)}."
            )
    elif isinstance(user_prompt, str):
        conversation.append({"role": "user", "content": user_prompt})
        if assistant_response:
            conversation.append({"role": "assistant", "content": assistant_response})
    elif isinstance(user_prompt, list):
        assistant_response = assistant_response or []
        if not (
            len(assistant_response) == len(user_prompt)
            or len(assistant_response) == len(user_prompt) - 1
        ):
            raise ValueError(
                "Inconsistent number of conversation turns: "
                f"user_prompt has {len(user_prompt)} turns, "
                f"assistant_response has {len(assistant_response)} turns."
            )
        for i, user_msg in enumerate(user_prompt):
            conversation.append({"role": "user", "content": user_msg})
            if i < len(assistant_response):
                conversation.append({"role": "assistant", "content": assistant_response[i]})
    if not conversation:
        raise ValueError(f"Got empty conversation.")
    return conversation


def get_transform_fn(transform_fn_id, **kwargs) -> Callable[[List[str]], List[str]]:
    stop_strings = kwargs.get("stop_strings")
    def coqa_after_generation_transform_fn(text_generated: List[str]):
        if stop_strings is None:
            return text_generated
        for i, sample in enumerate(text_generated):
            stop_at = len(sample)
            sliced_answer = sample
            for stop in stop_strings:
                if sample.endswith(stop):
                    stop_at = len(sample) - len(stop)
                    sliced_answer = sample[:stop_at]
                    break
            if sliced_answer.startswith("A: "):
                sliced_answer = sliced_answer.lstrip("A: ")
            sliced_answer = sliced_answer.rstrip()
            text_generated[i] = sliced_answer
        return text_generated
    if "coqa" in transform_fn_id:
        return coqa_after_generation_transform_fn
    raise NotImplementedError