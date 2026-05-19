from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from . import methods
from .models import LLM, EmbeddingModel
from .utils import extract_boxed


# ============================================================
# BATCH SIZE
# ============================================================

def compute_batch_size(cfg: Dict, model_name: str) -> int:
    base = cfg["global"]["base_batch_size"]
    model_scale = cfg["models"][model_name].get("batch_size_scale", 1.0)
    return max(1, int(base * model_scale))


# ============================================================
# TEST METHOD
# ============================================================

def run_method(
    cfg: Dict,
    method_name: str,
    model_name: str,
    llm: LLM,
    enable_logging: bool = True
):

    # -----------------------------
    # Load method
    # -----------------------------
    if not hasattr(methods, method_name):
        raise ImportError(f"Unknown method: {method_name}")

    method_class = getattr(methods, method_name)
    sig = inspect.signature(method_class)
    params = sig.parameters

    # -----------------------------
    # Embed model (if needed)
    # -----------------------------
    embed_model = None
    if "embed_model" in params:
        embed_model_name = cfg["methods"][method_name].get("embed_model_name", None)
        if embed_model_name:
            embed_model = EmbeddingModel(embed_model_name)

    # -----------------------------
    # Init kwargs
    # -----------------------------
    init_kwargs = {}

    if "llm" in params:
        init_kwargs["llm"] = llm

    if embed_model is not None and "embed_model" in params:
        init_kwargs["embed_model"] = embed_model

    # -----------------------------
    # Logging
    # -----------------------------
    if enable_logging and "logging" in params:
        Path(cfg["global"]["log_dir"]).mkdir(parents=True, exist_ok=True)

        init_kwargs["logging"] = str(
            Path(cfg["global"]["log_dir"])
            / f"log_{method_name}_{model_name.split('/')[-1]}_{int(time.time())}.txt"
        )

    # -----------------------------
    # Config injection
    # -----------------------------
    merged_cfg = {**cfg["global"], **cfg["methods"][method_name]}
    merged_cfg.pop("batch_size_scale", None)

    for k, v in merged_cfg.items():
        if k in params:
            init_kwargs[k] = v

    # -----------------------------
    # Instantiate method
    # -----------------------------
    method = method_class(**init_kwargs)

    # -----------------------------
    # Batch size
    # -----------------------------
    batch_size = compute_batch_size(cfg, model_name)

    # -----------------------------
    # Load data
    # -----------------------------
    test_df = pd.read_excel(cfg["global"]["test_data"])

    equations = test_df["equation"].tolist()
    solutions = test_df["solution"].tolist()

    # ✅ NEW: type column from dataset
    types = test_df["type"].tolist() if "type" in test_df.columns else ["unknown"] * len(test_df)

    train_path = cfg["global"].get("train_data", None)
    train_df = pd.read_excel(train_path) if train_path else None

    # -----------------------------
    # Fit
    # -----------------------------
    load_flag = cfg["methods"][method_name].get("load", True)
    if hasattr(method, "fit") and load_flag is False:
        train_eqs = train_df["equation"].tolist() if train_df is not None else []
        train_sols = train_df["solution"].tolist() if train_df is not None else []

        method.fit(train_eqs, train_sols, batch_size=batch_size)
        torch.cuda.empty_cache()

    # -----------------------------
    # Solve
    # -----------------------------
    raw_outputs = method.solve(equations, batch_size=batch_size)
    torch.cuda.empty_cache()

    # -----------------------------
    # Results
    # -----------------------------
    results = []

    for eq, out, gold, type_ in zip(equations, raw_outputs, solutions, types):

        response = out["response"]
        token_count = out["token_count"]
        avg_time = out["avg_time"]

        results.append({
            "equation": eq,
            "prediction": response,
            "final_answer": extract_boxed(response),
            "ground_truth": gold,
            "token_count": token_count,
            "avg_time": avg_time,
            "type": type_,
        })

    # -----------------------------
    # SAVE
    # -----------------------------
    Path(cfg["global"]["out_dir"]).mkdir(parents=True, exist_ok=True)

    output_path = Path(cfg["global"]["out_dir"]) / (
        f"results_{method_name}_{model_name.split('/')[-1]}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results