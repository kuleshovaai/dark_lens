from typing import Dict, Tuple, Type, List
from dataclasses import dataclass
from src.hallucination_detection.hallucination_detection import \
    HallucinationDetectionMethod

# from .haloscope.haloscope_detector import HaloscopeDetector
from .inside.inside import INSIDE
from .llm_check.llm_check import LLMCheck
from .max_prob.max_prob import MaxProb
from .mtopdiv.mtopdiv import MTopDiv
from .lumina.lumina import LUMINA
from .redeep.redeep import ReDeEP
from .dark_lens.dark_lens import DarkLens
from .neural_topology.neural_topology import NeuralTopology
from .perplexity.perplexity import Perplexity
from .selfcheck.selfcheck_nli import CustomSelfCheckNLI
from .semantic_entropy.semantic_entropy import SemanticEntropy
from .tokenwise_entropy.tokenwise_entropy import TokenwiseEntropy
from .sink_rate.sink_rate import SinkRate
from .sink_rate.calibrated_entropy import CalibratedEntropy
# from .lookback_lens.lookback_lens import LookbackLens
from .sar.sar import SAR
from .rauq.rauq import RAUQ
from .focus.focus import Focus
from .semantic_density.semantic_density import SemanticDensity
from .eigval_laplacian.eigval_laplacian import EigValLaplacian
from .hidden_proj.hidden_projection import NullSpaceProjection

from src.llm import wrappers

AVAILABLE_WRAPPERS = wrappers.__all__

METHOD_REGISTRY: Dict[str, Type[HallucinationDetectionMethod]] = {
    "inside": INSIDE,
    "llm_check": LLMCheck,
    "max_prob": MaxProb,
    "mtopdiv": MTopDiv,
    "neural_topology": NeuralTopology,
    "perplexity": Perplexity,
    "selfcheck": CustomSelfCheckNLI,
    "semantic_entropy": SemanticEntropy,
    "tokenwise_entropy": TokenwiseEntropy,
    "lumina": LUMINA,
    "redeep": ReDeEP,
    "dark_lens": DarkLens,
    "sink_rate": SinkRate,
    "sar": SAR,
    "rauq": RAUQ,
    "focus": Focus,
    "calibrated_entropy": CalibratedEntropy,
    "semantic_density": SemanticDensity,
    "eigval_laplacian": EigValLaplacian,
    "hidden_proj": NullSpaceProjection
}

@dataclass
class MethodDependency:
    needs_hidden_states: bool
    needs_attentions: bool
    needs_wrappers: List[str]
    
    def __post_init__(self):
        for wrapper in self.needs_wrappers:
            if wrapper not in AVAILABLE_WRAPPERS:
                raise ValueError(f"Wrapper {wrapper} is not available.")

METHODS_DEPENDENCIES: Dict[str, MethodDependency] = {
    # name: (needs_hidden_states, needs_attentions)
    "inside": MethodDependency(True, False, []),
    "llm_check": MethodDependency(False, True, []),
    "max_prob": MethodDependency(False, False, []),
    "mtopdiv": MethodDependency(False, True, []),
    "neural_topology": MethodDependency(True, True, []),
    "perplexity": MethodDependency(False, False, []),
    "selfcheck": MethodDependency(False, False, []),
    "semantic_entropy": MethodDependency(False, False, []),
    "tokenwise_entropy": MethodDependency(False, False, []),
    "lumina": MethodDependency(False, False, ["WrapperForAfterLayerLogits"]),
    "redeep": MethodDependency(True, True, ["WrapperForResidualStreamLogits"]),
    "dark_lens": MethodDependency(True, False, ["WrapperForResidualStreamLogits"]),
    "sink_rate": MethodDependency(False, True, []),
    "sar": MethodDependency(False, False, []),
    "rauq": MethodDependency(False, True, []),
    "focus": MethodDependency(False, True, []),
    "calibrated_entropy": MethodDependency(False, True, []),
    "semantic_density": MethodDependency(False, False, []),
    "eigval_laplacian": MethodDependency(False, False, []),
    "hidden_proj": MethodDependency(True, False, [])
}

def create_method(name: str, **kwargs) -> HallucinationDetectionMethod:
    if name not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        raise ValueError(
            f"Method '{name}' not found. "
            f"Available methods: {available}"
        )
    return METHOD_REGISTRY[name](**kwargs)

def get_output_requirements(name: str) -> MethodDependency:
    if name not in METHODS_DEPENDENCIES:
        raise ValueError(f"No requirements defined for method '{name}'")
    return METHODS_DEPENDENCIES[name]