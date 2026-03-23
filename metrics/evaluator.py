from __future__ import annotations

from typing import Dict, List

from utils.parsing import normalize_output

from metrics.bleu import bleu_score
from metrics.ast_error_size import ast_error_size
from metrics.ast_tree_distance import ast_tree_distance
from metrics.exact_match import exact_match


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    assert len(predictions) == len(references)

    bleu_vals = []
    ast_err_vals = []
    ast_tree_vals = []
    exact_vals = []

    for pred, ref in zip(predictions, references):
        pred_n = normalize_output(pred)
        ref_n = normalize_output(ref)

        bleu_vals.append(bleu_score(pred_n, ref_n))
        ast_err_vals.append(ast_error_size(pred_n, ref_n))
        ast_tree_vals.append(ast_tree_distance(pred_n, ref_n))

        exact_vals.append(exact_match(pred_n, ref_n))

    n = max(len(predictions), 1)

    return {
        "bleu": sum(bleu_vals) / n,
        "ast_error_size": sum(ast_err_vals) / n,
        "ast_tree_distance": sum(ast_tree_vals) / n,
        "exact_match": sum(exact_vals) / n,
    }