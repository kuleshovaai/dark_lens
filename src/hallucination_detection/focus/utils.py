import os
import numpy as np
import torch
import math
import dataclasses
from typing import Tuple, List
from spacy import Language
import pickle
from tqdm import tqdm
from torch.nn import NLLLoss
from datasets import load_dataset
from collections import defaultdict
import random
from loguru import logger as log
import spacy
from src.llm import LLM

def calcu_idf(
    model_name, path, idf_dataset, trust_remote_code, idf_seed, idf_dataset_size
):
    """
    Calculate inverse document frequency (IDF) scores for each token using a Hugging Face tokenizer
    and dataset. Results are saved to disk for reuse.

    Args:
       model_name (str): Path to the tokenizer model.
       path (str): File path to save computed IDF values.
       idf_dataset (str): Hugging Face dataset identifier for IDF computation.
       trust_remote_code (bool): Whether to trust remote code when loading the dataset.
       idf_seed (int): Random seed for dataset shuffling.
       idf_dataset_size (int): Max number of documents to use (-1 for all).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = load_dataset(idf_dataset, "sample", trust_remote_code=trust_remote_code)
    data = [d for d in dataset["train"]]
    rng = random.Random(idf_seed)
    rng.shuffle(data)

    if (idf_dataset_size > 0) and (idf_dataset_size < len(data)):
        data = rng.sample(data, idf_dataset_size)
        
    tokenizer = LLM(model_name=model_name).tokenizer
    document_frequency = defaultdict(int)
    offset = 1 if "facebook" in model_name else 0
    for doc in tqdm(data):
        tokenized_doc = tokenizer(doc["raw_content"])["input_ids"][offset:]
        unique_tokens = set(tokenized_doc)
        for token in unique_tokens:
            document_frequency[token] += 1

    total_documents = len(data)
    pickle.dump(
        np.array(
            [
                math.log(total_documents / (document_frequency[i] + 1))
                for i in range(len(tokenizer.vocab))
            ]
        ),
        open(path, "wb"),
    )


@dataclasses.dataclass
class IDFStats:
    """
    Container for IDF-related statistics and resources used in the Focus estimator.

    Attributes:
        token_idf (List): List of IDF values per token index.
        NER_type (List[str]): Named entity types considered important.
        pos_tag (List[str]): POS tags considered important.
        nlp (Language): Loaded spaCy NLP pipeline.
    """

    token_idf: List
    NER_type: List[str]
    pos_tag: List[str]
    nlp: Language


def load_idf(
    model_name: str,
    path: str,
    idf_dataset: str,
    trust_remote_code: bool,
    idf_seed: int,
    idf_dataset_size: int,
    spacy_path: str,
) -> IDFStats:
    """
    Load IDF statistics and spaCy model, computing IDF values if not already saved.

    Args:
        model_name (str): Tokenizer model name or path.
        path (str): Path to load or save the IDF file.
        idf_dataset (str): Dataset name used to calculate IDF.
        trust_remote_code (bool): Trust remote dataset loading code.
        idf_seed (int): Random seed for sampling.
        idf_dataset_size (int): Max number of samples to use for IDF.
        spacy_path (str): Name or path of the spaCy language model.

    Returns:
        IDFStats: Loaded or computed IDF statistics.
    """

    if not spacy.util.is_package(spacy_path):
        spacy.cli.download(spacy_path)
    if not os.path.exists(path):
        calcu_idf(
            model_name,
            path,
            idf_dataset,
            trust_remote_code,
            idf_seed,
            idf_dataset_size,
        )
    return IDFStats(
        token_idf=pickle.load(open(path, "rb")),
        NER_type=[
            "PERSON",
            "DATE",
            "ORG",
            "GPE",
            "NORP",
            "ORDINAL",
            "PRODUCT",
            "CARDINAL",
            "LOC",
            "FAC",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
            "TIME",
            "PERCENT",
            "MONEY",
            "QUANTITY",
        ],
        pos_tag=["NOUN", "NUM", "PROPN"],
        nlp=spacy.load(spacy_path),
    )


def entropy2(p):
    """
    Compute the entropy of a probability distribution using base-2 logarithm.

    Args:
        p (array-like): Probability distribution.

    Returns:
        float: Entropy value.
    """

    return torch.sum(
        -torch.where(p > 0, p * p.log2(), p.new([0.0])),
        dim=-1,
    ).numpy()


def token_level_focus_scores(
    response,
    response_ids,
    offset_mapping,
    log_probs,
    max_token_attn,
    idf: IDFStats,
    p: float,
    gamma: float,
) -> Tuple[List, List]:
    """
    Compute token-level Focus uncertainty scores and keyword masks based on
    attention, IDF, and linguistic signals (NER, POS).
    """
    loss_fct = NLLLoss(reduction="none")
    
    greedy_log_prob, attention_weight, greedy_token, greedy_text = log_probs, max_token_attn, response_ids, response
    
    sentence = idf.nlp(greedy_text)
    decodings = [response[s:e] for (s, e) in offset_mapping]
    span_index = 0
    prev_span_index = 0
    kw_mask = np.zeros_like(greedy_token, dtype=bool)

    for token_index, token in enumerate(decodings):
        span_index = prev_span_index
        while span_index < len(sentence) and (token.strip() not in sentence[span_index].text) and (
            sentence[span_index].text not in token.strip()
        ):
            span_index += 1
        if span_index == len(sentence):
            log.warning(
                f"Token `{token}` is not recognized. Most likely, this is due to misalignment between the text span and token positions. "
                "The corresponding keyword mask value will be set to 0."
            )
            continue
        else:
            prev_span_index = span_index
        span = sentence[span_index]
        if span.text not in idf.NER_type and (
            span.ent_type_ in idf.NER_type or span.pos_ in idf.pos_tag
        ):
            kw_mask[token_index] = True

    prob = np.exp(greedy_log_prob)
    mask = prob < p
    # only focus on keywords like NER
    prob[mask] = 0
    if prob.shape[-1] > len(idf.token_idf):
        prob[:, : len(idf.token_idf)] = (
            prob[:, : len(idf.token_idf)] * idf.token_idf
        )
    else:
        prob = prob * idf.token_idf
    prob = prob / prob.sum(axis=-1, keepdims=True)
    entropy = np.exp2(entropy2(prob))

    ll = loss_fct(torch.log(prob + 1e-5), greedy_token)
    hc = ll + entropy

    if not kw_mask.sum():
        return [], kw_mask
    # w(i,j) estimation and penalty estimation for a new hallucination score
    weight = np.zeros_like(attention_weight)
    attn_kw = attention_weight[kw_mask]
    if attn_kw.size == 0:
        return [], kw_mask

    # Безопасное нормализация
    if attn_kw.ndim == 1:
        weight[kw_mask] = attn_kw / (attn_kw.sum(keepdims=True) + 1e-6)
    else:
        weight[kw_mask] = attn_kw / (attn_kw.sum(axis=1, keepdims=True) + 1e-6)

    token_focus = []
    for i, token_weights in enumerate(weight):
        ue = hc[i]
        if len(token_focus):
            ue += (
                gamma
                * (np.array(token_focus) * token_weights[: len(token_focus)]).sum()
            )
        token_focus.append(ue)

    return token_focus, kw_mask
