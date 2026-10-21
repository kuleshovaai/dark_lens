import os
import hydra
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf
from src.evaluation import evaluate
from src.evaluation.process_metrics import process_metrics
from src.dataset import load_dataset
from src.hallucination_detection import create_method, get_output_requirements
from src.hallucination_detection.redeep.utils import resolve_copy_heads
from src.llm import LLM, LLMConfig, LLMOutputProvider
from src.llm import wrappers
from pathlib import Path
from huggingface_hub import login

load_dotenv()
login(os.getenv("HUGGING_FACE_API_KEY"))


def get_unembedding_matrix(llm_model: LLM) -> torch.Tensor:
    """Return HF lm_head weights on CPU for null-space SVD."""
    return llm_model.model.lm_head.weight.detach().cpu()


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: OmegaConf):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    hydra_cfg = HydraConfig.get()
    data_name: str = hydra_cfg.runtime.choices["data"]
    method_name: str = hydra_cfg.runtime.choices["method"]
    model_name: str = cfg["model_name"]

    cache_dir = Path(cfg["cache_dir"]) / data_name / model_name
    llm_model = LLM(model_name=model_name, device=cfg["device"], cache_dir=str(cache_dir))
    llm_config = LLMConfig(llm_model)

    method_kwargs = {
        **cfg["method"],
        "llm_config": llm_config,
        "cache_dir": str(cache_dir / method_name),
    }
    if method_name in ("dark_lens"):
        method_kwargs["unembedding_matrix"] = get_unembedding_matrix(llm_model)
    if method_name == "redeep":
        method_kwargs["copy_heads"] = resolve_copy_heads(
            llm_model,
            method_kwargs.get("copy_heads"),
            cache_path=str(cache_dir / "redeep" / "copy_heads.json"),
        )

    hal_detection_method = create_method(method_name, **method_kwargs)
    method_dependencies = get_output_requirements(method_name)

    outputs_provider = LLMOutputProvider(
        llm_model,
        batch_size=cfg["batch_size"],
        output_hidden_states=method_dependencies.needs_hidden_states,
        output_attentions=method_dependencies.needs_attentions,
        wrappers=[getattr(wrappers, wrapper_name) for wrapper_name in method_dependencies.needs_wrappers],
    )

    source_dir = cfg["data"]["source_dir"]
    file_path = str(Path(source_dir) / "annotated" / f"{data_name}_{model_name}.csv")
    dataset = load_dataset(file_path=file_path)

    generation_config = cfg["generation"]

    metrics = evaluate(
        dataset,
        hal_detection_method,
        llm_model,
        outputs_provider,
        generation_config,
        data_name,
        **cfg["evaluation"],
    )

    table_str, raw_table = process_metrics(metrics)
    logger.success(f"Results for cross validation on {data_name}")
    print(table_str)
    logger.info(f"final_test_auroc={raw_table['roc_auc'].loc['test']['mean']:.4f}")


if __name__ == "__main__":
    main()
