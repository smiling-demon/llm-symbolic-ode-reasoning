import sympy as sp
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def replace_constants(expr):
    """
    Normalizes symbolic constants in a SymPy expression.

    Replaces all symbols of the form C1, C2, C3, ... with a single symbol C.

    Args:
        expr: SymPy expression.

    Returns:
        SymPy expression with unified constants.
    """
    consts = [s for s in expr.free_symbols if s.name.startswith("C")]

    for c in consts:
        expr = expr.subs(c, sp.Symbol("C"))

    return expr


def sympy_to_text(expr):
    """
    Converts a SymPy expression into a stable textual representation.

    Args:
        expr: SymPy expression.

    Returns:
        str: Deterministic string representation.
    """
    expr = expr.as_ordered_terms()
    return str(expr)


def bleu_score(ground_truth, predicted_answer):
    """
    Computes BLEU score between two SymPy expressions.

    The comparison is performed after:
    - normalizing constants (C1, C2, ...)
    - converting expressions to ordered textual form

    Args:
        ground_truth: Reference SymPy expression.
        predicted_answer: Predicted SymPy expression.

    Returns:
        float: BLEU score in range [0, 1]. Returns 0.0 on failure.
    """
    try:
        gt = replace_constants(ground_truth)
        pr = replace_constants(predicted_answer)

        gt_tokens = sympy_to_text(gt).split()
        pr_tokens = sympy_to_text(pr).split()

        return sentence_bleu(
            [gt_tokens],
            pr_tokens,
            smoothing_function=SmoothingFunction().method1,
        )

    except Exception:
        return 0.0
