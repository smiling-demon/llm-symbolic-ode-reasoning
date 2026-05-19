from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

from .metrics import bleu_score, difference_error, substitution_error, llm_judge
from .utils import equation_to_sympy, solution_to_sympy


DEFAULT_METRICS = ["bleu", "difference_error", "substitution_error"]
ALL_METRICS = ["bleu", "difference_error", "substitution_error", "llm_judge"]


# =========================================================
# CLI
# =========================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ODE solution evaluation.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=Path)
    input_group.add_argument("--input_file", type=Path)

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output_dir", type=Path)
    output_group.add_argument("--output_file", type=Path)

    parser.add_argument("--time_limit", type=float, default=5.0)

    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        choices=ALL_METRICS,
    )

    # 🔥 NEW: parallel switch
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable multiprocessing over files",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Number of processes for parallel mode",
    )

    return parser.parse_args()


# =========================================================
# IO
# =========================================================

def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected list JSON")
    return data


def _save_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# SAFE WRAPPER
# =========================================================

def _safe(fn: Callable[..., Any], *args: Any):
    try:
        return fn(*args)
    except Exception:
        return None


# =========================================================
# CORE EVALUATION
# =========================================================

def _evaluate_item(item: dict[str, Any], metrics: list[str], time_limit: float) -> dict[str, Any]:
    equation = item["equation"]
    final_answer = item["final_answer"]
    ground_truth = item["ground_truth"]

    result = dict(item)

    gt_expr = None
    pred_expr = None
    eq_expr = None

    need_sol = any(m in metrics for m in ("bleu", "difference_error", "substitution_error"))
    need_eq = "substitution_error" in metrics

    if need_sol and final_answer and ground_truth:
        gt_expr = _safe(solution_to_sympy, ground_truth)
        pred_expr = _safe(solution_to_sympy, final_answer)

    if need_eq and equation:
        eq_expr = _safe(equation_to_sympy, equation)

    bleu = diff = sub = llm = None

    if "bleu" in metrics:
        bleu = _safe(bleu_score, gt_expr, pred_expr) if gt_expr and pred_expr else None
        result["bleu"] = bleu

    if "difference_error" in metrics:
        diff = _safe(difference_error, gt_expr, pred_expr, time_limit) if gt_expr and pred_expr else None
        result["difference_error"] = diff

    if "substitution_error" in metrics:
        sub = _safe(substitution_error, eq_expr, pred_expr, time_limit) if eq_expr and pred_expr else None
        result["substitution_error"] = sub

    if "llm_judge" in metrics:
        llm = _safe(llm_judge, ground_truth, final_answer)

    # accuracy rule
    result["accuracy"] = 1.0 if (
        llm == 1 or bleu == 1 or diff == 0 or sub == 0
    ) else 0.0

    return result


def _evaluate_file(input_path: Path, output_path: Path, metrics: list[str], time_limit: float):
    data = _load_json(input_path)

    evaluated = [
        _evaluate_item(item, metrics, time_limit)
        for item in tqdm(data, desc=f"Evaluating {input_path.name}")
    ]

    _save_json(output_path, evaluated)


# =========================================================
# PARALLEL WORKER
# =========================================================

def _process_file(task):
    in_path, out_path, metrics, time_limit = task
    _evaluate_file(in_path, out_path, metrics, time_limit)
    return in_path.name


def _evaluate_directory(
    input_dir: Path,
    output_dir: Path,
    metrics: list[str],
    time_limit: float,
    parallel: bool,
    workers: int,
):
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix == ".json"
    )

    tasks = [
        (f, output_dir / f.name, metrics, time_limit)
        for f in files
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    if not parallel:
        for t in tqdm(tasks, desc="Sequential processing"):
            _process_file(t)
        return

    print(f"Running parallel with {workers} workers on {len(tasks)} files")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_file, t) for t in tasks]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Parallel processing"):
            f.result()


# =========================================================
# MAIN
# =========================================================

def main():
    args = _parse_args()
    metrics = list(dict.fromkeys(args.metrics))

    if args.input_file:
        if not args.output_file:
            raise ValueError("--output_file required")

        _evaluate_file(args.input_file, args.output_file, metrics, args.time_limit)
        return

    if args.input_dir:
        if not args.output_dir:
            raise ValueError("--output_dir required")

        _evaluate_directory(
            args.input_dir,
            args.output_dir,
            metrics,
            args.time_limit,
            parallel=args.parallel,
            workers=args.workers,
        )
        return

    raise RuntimeError("No input provided")


if __name__ == "__main__":
    main()