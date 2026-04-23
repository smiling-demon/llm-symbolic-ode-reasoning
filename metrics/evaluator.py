from __future__ import annotations

from typing import Dict, List

from metrics.bleu import bleu_score
from metrics.ast_error_size import ast_error_size
from metrics.ast_tree_similarity import ast_tree_similarity
from metrics.exact_match import exact_match

from utils.parsing import extract_boxed


import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException()

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    timeout: float = 5.0,
) -> Dict[str, float]:
    assert len(predictions) == len(references)

    bleu_vals = []
    ast_err_vals = []
    ast_tree_vals = []
    exact_vals = []

    for pred, ref in zip(predictions, references):

        try:
            pred_expr = extract_boxed(pred)
        except Exception:
            pred_expr = None

        if not pred_expr:
            exact_vals.append(0)
            bleu_vals.append(0)
            ast_err_vals.append(1)
            ast_tree_vals.append(0)
            continue

        try:
            with time_limit(timeout):
                exact_vals.append(exact_match(pred_expr, ref))
        except TimeoutException:
            exact_vals.append(0)
        except Exception:
            exact_vals.append(0)

        # -------------------------
        # BLEU
        # -------------------------
        try:
            with time_limit(timeout):
                bleu_vals.append(bleu_score(pred_expr, ref))
        except TimeoutException:
            bleu_vals.append(0)
        except Exception:
            bleu_vals.append(0)

        try:
            with time_limit(timeout):
                ast_err_vals.append(ast_error_size(pred_expr, ref))
        except TimeoutException:
            ast_err_vals.append(1)
        except Exception:
            ast_err_vals.append(1)

        try:
            with time_limit(timeout):
                ast_tree_vals.append(ast_tree_similarity(pred_expr, ref))
        except TimeoutException:
            ast_tree_vals.append(0)
        except Exception:
            ast_tree_vals.append(0)

    n = max(len(predictions), 1)

    return {
        "bleu": sum(bleu_vals) / n,
        "ast_error_size": sum(ast_err_vals) / n,
        "ast_tree_similarity": sum(ast_tree_vals) / n,
        "exact_match": sum(exact_vals) / n,
    }
