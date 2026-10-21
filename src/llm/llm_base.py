from dataclasses import dataclass
from typing import List, Callable, Union, Optional, Any, Dict, cast, Iterable
from src.customtypes import LLMOutputs, LLMResponse, CausalLMOutputWithExtras
import torch
from dotenv import load_dotenv
from transformers import BatchEncoding
from transformers import AutoConfig
import types
from .llm_utils import (
    get_model_id,
    instantiate_model,
    instantiate_tokenizer,
    check_prompt_response_compatibility,
    get_conversation_dict,
    get_transform_fn,
)
from .caching_utils import LLMResponseCache

load_dotenv()


@dataclass
class LLMBase:
    """Base class for interacting with a LLM.

    This class handles model instantiation, generation of LLM outputs (hidden states, attentions),
    and generation of LLM responses based on various prompting strategies (chat templates, multi-turn).

    Attributes:
        model_name (str): The name of the pre-trained model to load. Defaults to "Qwen2.5-0.5B-Instruct".
        dtype (str): The data type for the model (e.g., "float16"). Defaults to "bfloat16".
        device (str): The device to load the model on (e.g., "cuda", "cpu"). Defaults to "cuda".
        attn_implementation (str): The attention implementation to use (e.g., "eager"). Defaults to "sdpa".
        use_cache (bool): Whether to cache functions outputs (currenly caches only LLM's responses). Defaults to True.
        cache_dir (str): The path to the stored cache. Defaults to "cache".
    """
    model_name: str = "Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    use_cache: bool = True
    cache_dir: str = "cache"

    def __repr__(self):
        return f"LLMBase(model_name={self.model_name}, dtype={self.dtype})"
    
    @property
    def cache(self):
        return self.cache_dir
    
    @cache.setter
    def cache(self, path):
        self.cache_dir = path
        if not self.use_cache:
            return
        wrapper = LLMResponseCache(cache_dir=path)
        wrapped = wrapper(self._original_generate_llm_responses)
        self.generate_llm_responses = types.MethodType(wrapped, self)
    
    def __post_init__(self):
        self._original_generate_llm_responses = self.generate_llm_responses.__func__
        model_id = get_model_id(self.model_name)
        self.config = AutoConfig.from_pretrained(model_id)
        self.cache = self.cache_dir

    @property
    def tokenizer(self):
        if "_tokenizer" in dir(self) and self._tokenizer is not None:
            return self._tokenizer
        self._tokenizer = instantiate_tokenizer(self.model_name)
        return self._tokenizer
    
    @property
    def model(self):
        if "_model" in dir(self) and self._model is not None:
            return self._model
        self._model = instantiate_model(
            self.model_name,
            self.dtype,
            self.device,
            self.attn_implementation,
        )
        return self._model

    @torch.no_grad()
    def generate_llm_outputs(
        self,
        prompt: List[str],
        answer: List[str],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        batch_size: int = 1,
    ) -> Iterable[LLMOutputs]:
        """Generates LLM outputs (logits, hidden states, attentions) for given prompts and answers.

        This method processes inputs in batches and yields the model outputs along with tokenized prompt and answer IDs.

        Args:
            prompt (List[str]): A list of input prompts.
            answer (List[str]): A list of corresponding answers.
            output_hidden_states (bool): Whether to return all hidden states. Defaults to True.
            output_attentions (bool): Whether to return all attention weights. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            move_to_cpu (bool): Whether to move the model outputs to CPU after generation. Defaults to True.

        Yields:
            Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
                A generator that yields a tuple containing:
                - output: The causal language model output (e.g., hidden states, attentions).
                - prompt_tokenized: Tokenized input IDs and attention mask for the prompts.
                - answer_tokenized: Tokenized input IDs and attention mask for the answers.
        """
        for batch_index in range(0, len(prompt), batch_size):
            prompt_batch = prompt[batch_index: batch_index + batch_size]
            answer_batch = answer[batch_index: batch_index + batch_size]
            prompt_text = self.construct_query(
                user_prompt=prompt_batch,
                apply_chat_template=False,
            )
            prompt_tokenized = self.tokenizer(
                prompt_text,
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            answer_text = self.construct_query(
                assistant_response=answer_batch,
                apply_chat_template=False,
                eos_token=self.tokenizer.eos_token
            )
            answer_tokenized = self.tokenizer(
                answer_text,
                padding=True,
                padding_side="right",
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            prompt_tokenized = cast(BatchEncoding, prompt_tokenized)
            answer_tokenized = cast(BatchEncoding, answer_tokenized)
            input_ids = torch.cat(
                [prompt_tokenized.input_ids, answer_tokenized.input_ids], dim=1
            ).to(self.device)
            attention_mask = torch.cat(
                [prompt_tokenized.attention_mask, answer_tokenized.attention_mask], dim=1
            ).to(self.device)
            if output_attentions:
                self.model.set_attn_implementation("eager")

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            self.model.set_attn_implementation(self.attn_implementation)
            output.past_key_values = None
            
            prompt_tokenized["special_tokens_mask"] = torch.zeros_like(prompt_tokenized.attention_mask)
            answer_tokenized["special_tokens_mask"] = torch.zeros_like(answer_tokenized.attention_mask)
            for token_id in self.tokenizer.all_special_ids:
                prompt_tokenized["special_tokens_mask"] += (prompt_tokenized.input_ids == token_id)
                answer_tokenized["special_tokens_mask"] += (answer_tokenized.input_ids == token_id)
            yield CausalLMOutputWithExtras(**output), prompt_tokenized, answer_tokenized

    @torch.no_grad()
    def generate_llm_responses(
        self,
        user_prompt: Union[List[str], List[List[str]]],
        assistant_response: Optional[Union[List[str], List[List[str]]]] = None,
        system_prompt: Optional[str] = None,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        batch_size: int = 1,
        apply_chat_template: bool = True,
        enable_thinking: bool = False,
        multi_turn_conversation: bool = False,
        stop_strings: Optional[List[str]] = None,
        after_generation_transform_fn_id: str = None,
        tools: Optional[Any] = None,
        documents: Optional[Any] = None,
        **kwargs,
    ) -> Iterable[LLMResponse]:
        """Generates LLM responses based on provided messages.

        This method supports various generation configurations including chat templates,
        multi-turn conversations, and custom post-processing of generated text.

        Args:
            user_prompt : List[str] | List[List[str]]
                User input(s). Can be:
                - a list of strings (batched prompts or multi-turn conversation)
                - a list of list of strings (batched multi-turn conversations)
            assistant_response : List[str] | List[List[str]] | None
                Assistant response(s) aligned with `user_prompt`.
            system_prompt : str | None
                Optional system instruction prepended to each prompt.
            num_return_sequences (int): Number of variants to generate for each sample. Defaults to 1.
            do_sample (bool): Whether to apply sampling. Defaults to False.
            max_new_tokens (int): Maximum number of tokens to generate. Defaults to 512.
            temperature (float): Temperature applied during sampling. Defaults to 1.0.
            batch_size (int): Batch size. Defaults to 1.
            apply_chat_template (bool): Whether to format inputs using the tokenizer's chat template. Defaults to True.
            enable_thinking (bool): Whether to enable 'thinking' capability in chat templates. Defaults to False.
            multi_turn_conversation (bool): Whether to treat the input as a multi-turn conversation. Defaults to False.
            stop_strings (Optional[List[str]]): A list of strings that, if generated, will stop the generation. Defaults to None.
            after_generation_transform_fn (Optional[Callable[[List[str]], List[str]]]):
                An optional function to apply to the generated text after decoding. Defaults to None.
            tools (Optional[Any]): Tools to be passed to the chat template for tool-augmented generation. Defaults to None.
            documents (Optional[Any]): Documents to be passed to the chat template for RAG. Defaults to None.
            **kwargs: Additional arguments passed to a model's generate method.

        Yields:
            Generator[Dict[str, Any], None, None]: A generator that yields dictionaries with:
                - "prompt": The original prompt text used for generation.
                - "response": A list of generated responses for that prompt.
        """
        if after_generation_transform_fn_id is not None:
            after_generation_transform_fn = get_transform_fn(
                after_generation_transform_fn_id,
                stop_strings=stop_strings,
            )
        else:
            after_generation_transform_fn = lambda x: x
        for batch_index in range(0, len(user_prompt), batch_size):
            user_prompt_batch = user_prompt[batch_index: batch_index + batch_size]
            if assistant_response is None or all(map(lambda x: x is None, assistant_response)):
                assistant_response_batch = None
            else:
                assistant_response_batch = assistant_response[batch_index: batch_index + batch_size]
            query = self.construct_query(
                user_prompt=user_prompt_batch,
                assistant_response=assistant_response_batch,
                system_prompt=system_prompt,
                apply_chat_template=apply_chat_template,
                tools=tools,
                documents=documents,
                add_generation_prompt=True,
                continue_final_message=False,
                enable_thinking=enable_thinking,
                multi_turn_conversation=multi_turn_conversation,
                bos_token=self.tokenizer.bos_token if self.tokenizer.bos_token is not None else ""
            )
            query_tokenized = self.tokenizer(
                query,
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                return_tensors="pt",
            )
            query_tokenized = cast(BatchEncoding, query_tokenized)
            query_tokenized = query_tokenized.to(self.model.device)
            prompt_len = query_tokenized.input_ids.shape[1]
            tokens_generated = self.model.generate(
                **query_tokenized, #type: ignore
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                stop_strings=stop_strings,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                tokenizer=self.tokenizer,
                **kwargs,
            )[:, prompt_len:].to('cpu')
            text_generated = self.tokenizer.batch_decode(
                tokens_generated, skip_special_tokens=True
            )
            if after_generation_transform_fn_id is not None:
                text_generated = after_generation_transform_fn(text_generated)
            for i in range(0, len(text_generated), num_return_sequences):
                yield LLMResponse(
                    prompt=query[i // num_return_sequences],
                    response=text_generated[i: i + num_return_sequences],
                )

    def construct_query(
        self,
        user_prompt: Optional[Union[str, List[str], List[List[str]]]] = None,
        assistant_response: Optional[Union[str, List[str], List[List[str]]]] = None,
        system_prompt: Optional[str] = None,
        apply_chat_template: bool = True,
        tools: Optional[List[Union[dict, Callable]]] = None,
        documents: Optional[List[Dict[str, str]]] = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        enable_thinking: bool = False,
        multi_turn_conversation: bool = False,
        bos_token: str = "",
        eos_token: str = "",
        sep_token: str = "",
        **kwargs: Any,
    ) -> Union[List[str], str]:
        """
        Construct text prompts (optionally using a chat template) and tokenize them.

        This method supports:
        - single-turn and multi-turn conversations
        - batched inputs
        - chat-template-based formatting or manual BOS/EOS concatenation
        - generation-ready prompts

        Parameters
        ----------
        user_prompt : str | List[str] | List[List[str]] | None
            User input(s). Can be:
            - a single string
            - a list of strings (batched prompts or multi-turn conversation)
            - a list of list of strings (batched multi-turn conversations)
        assistant_response : str | List[str] | List[List[str]] | None
            Assistant response(s) aligned with `user_prompt`.
        system_prompt : str | None
            Optional system instruction prepended to each prompt.
        apply_chat_template : bool
            Whether to format inputs using `tokenizer.apply_chat_template`. Defaults to True.
        tools : list[dict | Callable] | None
            Optional tool definitions passed to the chat template. Defaults to None.
        documents : list[dict[str, str]] | None
            Optional documents passed to the chat template. Defaults to None.
        add_generation_prompt : bool
            If apply_chat_template is True, argument passed to the chat template.
            Else if False, append EOS when no assistant response is provided. Defaults to True.
        continue_final_message : bool
            Whether to continue the final assistant message in chat templates. Defaults to False.
        enable_thinking : bool
            Whether to enable thinking in chat templates. Defaults to False.
        multi_turn_conversation : bool
            Interpret a flat list of strings as a single multi-turn conversation.
            If `apply_chat_template` is False, casted to False. Defaults to False.
        bos_token : str
            Beginning-of-sequence token used when `apply_chat_template` is False. Defaults to "".
        eos_token : str
            End-of-sequence token used when `apply_chat_template` is False. Defaults to "".
        sep_token : str
            Separator token used when `apply_chat_template` is False. Defaults to "".
        **kwargs
            Additional arguments.

        Returns
        -------
        Union[List[str], str]:
            - text: The raw text string or list of strings after template application/concatenation.
        """
        if isinstance(user_prompt, list) and len(user_prompt) == 0:
            user_prompt = None
        if isinstance(assistant_response, list) and len(assistant_response) == 0:
            assistant_response = None
        check_prompt_response_compatibility(user_prompt, assistant_response)
        if apply_chat_template is False:
            multi_turn_conversation = False
        is_batched = (
            isinstance(user_prompt, list) and len(user_prompt) > 0 and (
                not multi_turn_conversation
                or not isinstance(user_prompt[0], str)
            )
        )
        if (
            is_batched
            and isinstance(user_prompt, list)
            and assistant_response is not None
            and len(assistant_response) != len(user_prompt)
        ):
            raise ValueError(
                "Inconsistent number of samples: "
                f"user_prompt has {len(user_prompt)} samples, "
                f"assistant_response has {len(assistant_response)} samples."
            )
        if apply_chat_template:
            if is_batched:
                user_prompt = cast(list, user_prompt)
                conversation = []
                for idx in range(len(user_prompt)):
                    conversation.append(
                        get_conversation_dict(
                            user_prompt[idx],
                            assistant_response[idx] if assistant_response is not None else None,
                            system_prompt,
                        )
                    )
            else:
                user_prompt = cast(Union[None, str, List[str]], user_prompt)
                assistant_response = cast(Union[None, str, List[str]], assistant_response)
                conversation = get_conversation_dict(
                    user_prompt, assistant_response, system_prompt
                )
            text = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tools=tools,
                documents=documents,
                continue_final_message=continue_final_message,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                tokenize=False,
                **kwargs,
            )
        else:
            if user_prompt is None:
                if isinstance(assistant_response, list):
                    text = []
                    for resp in assistant_response:
                        if not isinstance(resp, str):
                            raise TypeError(
                                "When `apply_chat_template` is False, `assistant_response` must be List[str], str or None. "
                                f"Got assistant_response of type List[{type(assistant_response)}]."
                            )
                        text.append(resp + eos_token)
                elif isinstance(assistant_response, str):
                    text = assistant_response + eos_token
                else:
                    text = bos_token
                    text += system_prompt or ""
                    text += eos_token
            elif isinstance(user_prompt, str):
                text = bos_token
                if system_prompt is not None:
                    text += system_prompt + sep_token
                text += user_prompt
                if isinstance(assistant_response, str):
                    text += sep_token + assistant_response
                text += eos_token
            else:
                text = []
                user_prompt = cast(List[str], user_prompt)
                assistant_response = cast(Union[None, List[str]], assistant_response)
                for idx in range(len(user_prompt)):
                    if not isinstance(user_prompt[idx], str):
                        raise TypeError(
                            "When `apply_chat_template` is False, `user_prompt` must be List[str], str or None. "
                            f"Got user_prompt of type List[{type(user_prompt)}]."
                        )
                    entry = bos_token
                    if system_prompt is not None:
                        entry += system_prompt + sep_token
                    entry += user_prompt[idx]
                    if assistant_response is not None:
                        entry += sep_token + assistant_response[idx]
                    entry += eos_token
                    text.append(entry)
        return text


if __name__ == "__main__":
    llm = LLMBase(
        model_name="Qwen2.5-7B-Instruct",
        device="cuda:2",
        dtype="bfloat16",
        attn_implementation="sdpa",
        use_cache=True,
    )
    gen = llm.generate_llm_responses(user_prompt=["Say one", "Say two","Say three", "Say four", "Say five", "Say six", "Say seven"], do_sample=True, num_return_sequences=5)
    for output in gen:
        print(output)
