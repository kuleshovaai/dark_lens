from typing import Iterable, List, Tuple, Optional, TypeVar
from src.llm.llm_base import LLMBase
from src.customtypes import LLMResponse, CausalLMOutputWithExtras
from src.batching_utils import LLMOutputsWithQuery
from src.llm.wrappers import Wrapper


T = TypeVar('T', bound=Wrapper)

class LLMOutputProvider:
    def __init__(
        self,
        llm: LLMBase,
        batch_size: int = 1,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        wrappers: Optional[List[T]] = None,
    ):
        self.llm = llm
        self.batch_size = batch_size
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        instantiated = []
        for wrapper in wrappers or []:
            if isinstance(wrapper, type):
                instantiated.append(wrapper(llm.model))
            else:
                instantiated.append(wrapper)
        self.wrappers = instantiated

    def batch_stream(self, stream: Iterable[LLMResponse]) -> Iterable[Tuple[List, ...]]:
        buffer = {
            "ids": [],
            "counts": [],
            "prompts": [],
            "responses": []
        }
        for sample_id, item in enumerate(stream):
            responses = item["response"]
            num_seq = len(responses)
            for res in responses:
                buffer["ids"].append(sample_id)
                buffer["counts"].append(num_seq)
                buffer["prompts"].append(item["prompt"])
                buffer["responses"].append(res)
                if len(buffer["ids"]) == self.batch_size:
                    yield (
                        buffer["ids"],
                        buffer["counts"],
                        buffer["prompts"],
                        buffer["responses"]
                    )
                    buffer = {k: [] for k in buffer}
        if buffer["ids"]:
            yield (
                buffer["ids"],
                buffer["counts"],
                buffer["prompts"],
                buffer["responses"]
            )

    def __call__(
        self,
        response_stream: Iterable[LLMResponse],
    ) -> Iterable[LLMOutputsWithQuery]:
        for batch in self.batch_stream(response_stream):
            ids, counts, prompts, responses = batch
            for w in self.wrappers:
                w.enable()
            for outputs in self.llm.generate_llm_outputs(
                prompt=prompts,
                answer=responses,
                batch_size=self.batch_size,
                output_hidden_states=self.output_hidden_states,
                output_attentions=self.output_attentions,
            ):
                llm_forward_outputs, prompt_tokens, response_tokens = outputs
                wrappers_outputs_dict = {}
                for w in self.wrappers:
                    wrappers_outputs_dict.update(w.collect())
                llm_forward_outputs = CausalLMOutputWithExtras(**llm_forward_outputs, **wrappers_outputs_dict)
                yield LLMOutputsWithQuery(
                    sample_ids=ids,
                    num_seqs=counts,
                    prompts=prompts,
                    responses=responses,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    outputs=llm_forward_outputs,
                )
            for w in self.wrappers:
                w.disable()
