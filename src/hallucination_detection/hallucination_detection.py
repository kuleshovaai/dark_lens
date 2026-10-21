from typing import Optional, List, TypeVar, Generic, Iterable, Literal, ClassVar, Dict, Set, Any
from collections import OrderedDict
from abc import ABC, abstractmethod
import types
import numpy as np
from loguru import logger
from sklearn.metrics import f1_score
from src.customtypes import LLMResponse, LLMOutputs
from src.batching_utils import LLMOutputsWithQuery
from src.llm import LLMOutputProvider
from .caching_utils import HalDetTransformCache
from src.batching_utils import concat_outputs


class HallucinationDetectionMethod(ABC):
    """Abstract class for specifying method signatures and interface of the general hallucination detection methods."""
    method_type: ClassVar[Literal["black_box", "white_box"]]

    def __init__(self):
        self.best_thr: Optional[float] = None

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Dynamically creates and returns config of a hallucination detection method."""
        pass

    @abstractmethod
    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: List[List[float]],
        y_val: List[int],
    ) -> "HallucinationDetectionMethod":
        """Fit the detection model on training data."""
        pass

    @abstractmethod
    def predict_score(self, X: List[List[float]]) -> List[float]:
        """Predict probability of hallucination for samples from X."""
        pass

    @abstractmethod
    def reset(self):
        """For supervised models - return the model to the untrained state."""
        pass

    def set_threshold(self, thr: float) -> None:
        self.best_thr = thr

    def predict(self, X: List[List[float]]) -> List[int]:
        """Perform inference and return binary predictions based on the fitted threshold.

        Parameters
        ----------
        X : List[T]
            Input data.

        Returns
        -------
        prediction : List[int]
            Binary predictions (0 or 1) based on the threshold.

        """
        best_thr = self.best_thr
        if best_thr is None:
            raise ValueError("Threshold is not fitted. Call 'fit_threshold' first.")
        scores = self.predict_score(X)
        prediction = list(map(lambda s: int(s >= best_thr), scores))
        return prediction

    @staticmethod
    def get_threshold(val_scores: List[float], val_labels: List[int]) -> float:
        """Get the optimal threshold that maximizes F1 score on the training set.

        Parameters
        ----------
        val_scores : List[float]
            Predicted scores.
        val_labels : List[float]
            True labels.

        Returns
        -------
        float
            The best threshold that maximizes the F1 score.

        """
        max_f1 = 0
        max_thr = 0
        val_scores_np = np.asarray(val_scores)
        val_labels_np = np.asarray(val_labels)
        thresholds = np.linspace(np.min(val_scores_np), np.max(val_scores_np), 1000)
        for thr in thresholds:
            predictions = (val_scores_np > thr).astype(int)
            f1 = f1_score(val_labels_np, predictions)
            if f1 > max_f1:
                max_thr = thr
                max_f1 = f1
        logger.info(f'Selected threshold: {max_thr} with F1 score: {max_f1}')
        return max_thr

class WhiteBoxHalDetMethod(HallucinationDetectionMethod, ABC):
    method_type: ClassVar[Literal["black_box", "white_box"]] = "white_box"

    def __init__(self, batch_size: int = 1, use_cache: bool = True, cache_dir: str = "cache/"):
        super().__init__()
        self.batch_size: int = batch_size
        self.use_cache: bool = use_cache
        self.cache_dir: str = cache_dir
        self.cache = cache_dir
        
    @property
    def cache(self):
        return self.cache_dir
    
    @cache.setter
    def cache(self, path):
        self.cache_dir = path
        if not hasattr(self, "_original_transform"):
            self._original_transform = self.transform.__func__
        if self.use_cache:
            wrapper = HalDetTransformCache(cache_dir=path)
            wrapped = wrapper(self._original_transform)
            self.transform = types.MethodType(wrapped, self)

    @abstractmethod
    def get_score(self, queries_batch: List[LLMResponse], outputs_batch: List[LLMOutputs]) -> List[List[float]]:
        """Transform raw input data into model-ready representations."""
        pass

    @staticmethod
    def concat_outputs_with_query(
        *outputs_with_query: "LLMOutputsWithQuery", 
        pad_token_id: int,
    ) -> "LLMOutputsWithQuery":
        sample_ids = []
        num_seqs = []
        prompts = []
        responses = []
        prompt_tokens = []
        response_tokens = []
        outputs = []
        for output_with_query in outputs_with_query:
            sample_ids.extend(output_with_query.sample_ids)
            num_seqs.extend(output_with_query.num_seqs)
            prompts.extend(output_with_query.prompts)
            responses.extend(output_with_query.responses)
            prompt_tokens.append(output_with_query.prompt_tokens)
            response_tokens.append(output_with_query.response_tokens)
            outputs.append(output_with_query.outputs)
        outputs, prompt_tokens, response_tokens = concat_outputs(*zip(outputs, prompt_tokens, response_tokens), pad_token_id=pad_token_id)
        return LLMOutputsWithQuery(
            sample_ids=sample_ids,
            num_seqs=num_seqs,
            prompts=prompts,
            responses=responses,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            outputs=outputs,
        )

    def transform(
        self,
        stream: Iterable[LLMResponse],
        outputs_provider: LLMOutputProvider,
    ) -> Iterable[List[List[float]]]:
        outputs_stream = outputs_provider(stream)
        buffers: OrderedDict[int, List[LLMOutputsWithQuery]] = OrderedDict()
        num_of_prepared_samples = 0
        expected: Dict[int, int] = {}
        queries_batch = []
        outputs_batch = []
        ready: Set[int] = set()
        for outputs in outputs_stream:
            sample_ids = outputs.sample_ids
            num_seqs = outputs.num_seqs
            n = len(sample_ids)
            start = 0
            while start < n:
                end = start + 1
                sid = sample_ids[start]
                while end < n and sample_ids[end] == sid:
                    end += 1
                if sid not in buffers:
                    buffers[sid] = [outputs[start:end]]
                    expected[sid] = num_seqs[start]
                else:
                    buffers[sid].append(outputs[start:end])
                buffer_sid_length = sum([len(elm) for elm in buffers[sid]])
                if (buffer_sid_length == expected[sid] and sid not in ready):
                    ready.add(sid)
                    num_of_prepared_samples += 1
                start = end
            while num_of_prepared_samples > 0:
                sid, full = buffers.popitem(last=False)
                full = self.concat_outputs_with_query(*full, pad_token_id=outputs_provider.llm.tokenizer.pad_token_id)
                expected.pop(sid)
                ready.remove(sid)
                query = LLMResponse(
                    prompt=full.prompts[0],
                    response=full.responses,
                )
                queries_batch.append(query)
                llm_outputs = (full.outputs, full.prompt_tokens, full.response_tokens)
                outputs_batch.append(llm_outputs)
                num_of_prepared_samples -= 1
                if len(queries_batch) == self.batch_size:
                    yield self.get_score(queries_batch, outputs_batch)
                    queries_batch = []
                    outputs_batch = []
        if len(queries_batch) > 0:
            yield self.get_score(queries_batch, outputs_batch)

    @abstractmethod
    def fit_hyperparameters(
        self,
        stream: Iterable[LLMResponse],
        outputs_provider: LLMOutputProvider,
        y_val: List[int],
    ):
        """Perform hyperparameter tuning on the validation set if needed."""
        pass


class BlackBoxHalDetMethod(HallucinationDetectionMethod, ABC):
    method_type: ClassVar[Literal["black_box", "white_box"]] = "black_box"

    def __init__(self, batch_size: int = 1, use_cache: bool = True, cache_dir: str = "cache/"):
        super().__init__()
        self.batch_size: int = batch_size
        self.use_cache: bool = use_cache
        self.cache_dir: str = cache_dir
        self.cache = cache_dir
        
    @property
    def cache(self):
        return self.cache_dir
    
    @cache.setter
    def cache(self, path):
        self.cache_dir = path
        if not hasattr(self, "_original_transform"):
            self._original_transform = self.transform.__func__
        if self.use_cache:
            wrapper = HalDetTransformCache(cache_dir=path)
            wrapped = wrapper(self._original_transform)
            self.transform = types.MethodType(wrapped, self)

    @abstractmethod
    def get_score(self, queries_batch: List[LLMResponse]) -> List[List[float]]:
        """Transform raw input data into model-ready representations."""
        pass

    def transform(
        self,
        stream: Iterable[LLMResponse],
    ) -> Iterable[List[List[float]]]:
        queries_batch = []
        for sample in stream:
            queries_batch.append(sample)
            if len(queries_batch) == self.batch_size:
                yield self.get_score(queries_batch)
                queries_batch = []
        if len(queries_batch) > 0:
            yield self.get_score(queries_batch)

    @abstractmethod
    def fit_hyperparameters(
        self,
        stream: Iterable[LLMResponse],
        y_val: List[int],
    ):
        """Perform hyperparameter tuning on the validation set if needed."""
        pass
