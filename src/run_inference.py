from __future__ import annotations

import argparse
import gc
import yaml
import torch
from typing import Dict, List

from .run_method import run_method
from .models import LLM


# ============================================================
# MEMORY
# ============================================================

def clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ============================================================
# EXPERIMENT LOOP
# ============================================================

import time


def run_experiments(
    cfg: Dict,
    methods: List[str],
    models: List[str],
    enable_logging: bool = True
):
    for model_name in models:

        print("\n" + "=" * 30)
        print(f"MODEL: {model_name}")
        print("=" * 30 + "\n")

        llm = LLM(model_name)

        for method_name in methods:

            print(f"\n[START] {method_name} | {model_name}")
            start_time = time.perf_counter()

            run_method(
                cfg=cfg,
                method_name=method_name,
                model_name=model_name,
                llm=llm,
                enable_logging=enable_logging
            )

            end_time = time.perf_counter()

            print(
                f"[DONE ] {method_name} | {model_name} | "
                f"time: {end_time - start_time:.2f}s"
            )

        del llm
        clear_memory()


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_METHODS = [
    "Baseline",
    "ChainOfThought",
    "LeastToMost",
    "TreeOfThought",
    "ReasoningBank",
    "RecursiveSelfAggregation",
    "RSAWithToT",
    "RSAWithReasoningBank",
]

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to run (default: all)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to run (default: 3B only)",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override cfg['global']['base_batch_size']",
    )

    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable logging",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

        if args.batch_size is not None:
            cfg["global"]["base_batch_size"] = args.batch_size

    methods = args.methods if args.methods is not None else DEFAULT_METHODS
    models = args.models if args.models is not None else DEFAULT_MODELS

    run_experiments(
        cfg=cfg,
        methods=methods,
        models=models,
        enable_logging=not args.no_log,
    )
