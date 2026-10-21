import os
from typing import Iterable, Union, Tuple, List, Dict, cast
import copy
import joblib
import numpy as np
import pandas as pd
import json
from loguru import logger
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score, confusion_matrix) # Added confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from src.customtypes import LLMResponse
from src.llm import LLMOutputProvider, LLM
from src.hallucination_detection.hallucination_detection import \
    WhiteBoxHalDetMethod, BlackBoxHalDetMethod, HallucinationDetectionMethod
from src.evaluation.kfold3_split import KFold3


def get_stream(
    df: pd.DataFrame,
    llm: LLM,
    generation_config: dict,
) -> Iterable[LLMResponse]:
    """to be modified in the future
    Returns a stream that is gonna be used like:
    for features in hal_detection_method.transform(outputs_stream):
        ...
    """
    if generation_config["num_return_sequences"] > 1:
        llm_stream = llm.generate_llm_responses(
            user_prompt=df["prompt"].tolist(),
            assistant_response=df["response_history"].tolist(),
            **generation_config,
        )
        generation_config = dict(**generation_config)
        generation_config["num_return_sequences"] = 1
        def generator():
            for prompt, generated_response, responses, llm_query in zip(
                df["prompt"].tolist(),
                df["generated_response"].tolist(),
                df["response_history"].tolist(),
                llm_stream,
            ):
                yield LLMResponse(prompt=llm_query["prompt"], response=[generated_response] + llm_query["response"])
    else:
        def generator():
            for prompt, generated_response, responses in zip(
                df["prompt"].tolist(),
                df["generated_response"].tolist(),
                df["response_history"].tolist(),
            ):
                query = llm.construct_query(
                    user_prompt=prompt,
                    assistant_response=responses,
                    bos_token=llm.tokenizer.bos_token if llm.tokenizer.bos_token is not None else "",
                    **generation_config
                )
                query = cast(str, query)
                if isinstance(generated_response, str):
                    yield LLMResponse(prompt=query, response=[generated_response])
                else:
                    yield LLMResponse(prompt=query, response=generated_response)
    return generator()

def evaluate(
    df: pd.DataFrame,
    hal_detection_method: WhiteBoxHalDetMethod | BlackBoxHalDetMethod,
    llm_model: LLM,
    outputs_provider: LLMOutputProvider,
    generation_config: dict,
    data_name: str,
    k: int = 5,
    seed: int = 42,
    val_size: Union[float, int] = 0.15,
    test_size: float = 0.2,
    tune_hyperparameters: bool = True,
    pretrained: bool = False,
) -> List[Dict[str, str | float]]:
    
    result = []
    method_name = hal_detection_method.__class__.__name__

    metrics_path = f"cache/metrics/{data_name}/{llm_model.model_name}.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data[method_name] = []

    np.random.seed(42)
    
    X = df.drop(columns="hallucination")
    y = df["hallucination"].values
    
    context_series = df["context"].astype("object").copy()
    context_series = context_series.where(
        context_series.notna(),
        "__no_context__" + context_series.index.astype(str),
    )
    groups = context_series.to_numpy()

    skf = KFold3(
        n_splits=k,
        val_size=val_size,
        test_size=test_size,
        shuffle=True,
        random_state=seed,
    )

    cm_csv_data = []
    for train_idxs, val_idxs, test_idxs in tqdm(skf.split(X, y, groups), desc=f"Running KFold evaluation", total=k, unit="folds"):
        y_train = y[train_idxs]
        y_val = y[val_idxs]
        y_test = y[test_idxs]

        full_stream = get_stream(X, llm_model, generation_config)
        val_stream = get_stream(X.iloc[val_idxs], llm_model, generation_config)

        if tune_hyperparameters:
            if isinstance(hal_detection_method, WhiteBoxHalDetMethod):
                hal_detection_method.fit_hyperparameters(val_stream, outputs_provider=outputs_provider, y_val=y_val)
            else:
                hal_detection_method.fit_hyperparameters(val_stream, y_val)

        X_transformed = []
        if isinstance(hal_detection_method, WhiteBoxHalDetMethod):
            features_stream = hal_detection_method.transform(stream=full_stream, outputs_provider=outputs_provider)
        else:
            features_stream = hal_detection_method.transform(stream=full_stream)
            
        pbar = tqdm(
            desc="Haldet method transform",
            total=len(X),
            leave=False,
        )
            
        for features in features_stream:
            X_transformed.extend(features)
            pbar.update(len(features))
        pbar.close()

        X_train = [X_transformed[i] for i in train_idxs]
        X_val = [X_transformed[i] for i in val_idxs]
        X_test = [X_transformed[i] for i in test_idxs]
        
        if not pretrained:
            hal_detection_method.reset()
            hal_detection_method.fit(X_train, y_train, X_val, y_val)
            X_val_scores = hal_detection_method.predict_score(X_val)
            thr = hal_detection_method.get_threshold(X_val_scores, y_val)
            hal_detection_method.set_threshold(thr)

        # Evaluate on train, val, and test sets
        # Note: Added split_idxs to zip to keep track of original dataframe rows for filtering
        for dataset, X_split, y_split, split_idxs in zip(
            ["train", "val", "test"], 
            [X_train, X_val, X_test], 
            [y_train, y_val, y_test],
            [train_idxs, val_idxs, test_idxs]
        ):
            if len(y_split) == 0:
                logger.info(f"Skipping empty {dataset} split")
                continue

            # Predict probabilities and classes
            y_pred_score = hal_detection_method.predict_score(X_split)
            y_pred = hal_detection_method.predict(X_split)
            y_split = cast(List[float], y_split)
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_split, y_pred_score)
            f1 = f1_score(y_split, y_pred)

            logger.info(f"ROCAUC on {dataset} is {roc_auc}")

            # Generate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_split, y_pred_score)

            metrics_current = {
                "dataset": dataset,
                "roc_auc": roc_auc,
                "f1": f1,
                "pr_auc": auc(recall, precision),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "seed": seed,
            }

            if dataset == "test":
                data[method_name].append(roc_auc)
            
            if dataset == "train":
                # Fetch original text, force string, and lowercase
                gen_resp_series = X["generated_response"].iloc[split_idxs].astype(str).str.lower()
                
                # Filter for "yes" or "no" as distinct words
                binary_mask = gen_resp_series.str.contains(r'\b(?:yes|no)\b', regex=True).values
                
                y_true_bin = np.array(y_split)[binary_mask]
                y_pred_bin = np.array(y_pred)[binary_mask]
                
                if len(y_true_bin) > 0:
                    # labels=[0, 1] ensures it always returns a 2x2 matrix (even if a class is missing)
                    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
                    logger.info(f"Train set Confusion Matrix (Responses containing 'yes'/'no'):\n{cm}")
                    
                    # Unpack the 2x2 matrix
                    tn, fp, fn, tp = cm.ravel()
                    
                    # Save to our list (Assuming your loop variable is named `fold`)
                    # If your loop variable is named `i` or something else, change `fold` accordingly.
                    cm_csv_data.append({ 
                        "method" : method_name,
                        "TN": tn,
                        "FP": fp,
                        "FN": fn,
                        "TP": tp,
                        "Total_Binary_Samples": len(y_true_bin)
                    })
                else:
                    logger.info("No responses containing 'yes' or 'no' found in this train fold.")

            result.append(metrics_current)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(data, f)

    # After your k-fold loop concludes...
    if cm_csv_data:
        cm_df = pd.DataFrame(cm_csv_data)
        csv_filename = f"train_binary_confusion_matrices_{llm_model.model_name}.csv" # Change to your actual existing CSV name
        
        # Check if file exists to determine if we should write headers
        file_exists = os.path.isfile(csv_filename)
        
        # mode='a' appends to the file. 
        # header=not file_exists ensures we don't write headers in the middle of the document
        cm_df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
        
        logger.info(f"Appended confusion matrices to {csv_filename}")

    return result
