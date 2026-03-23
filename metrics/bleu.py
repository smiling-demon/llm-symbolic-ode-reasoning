from __future__ import annotations

import math
import re
from collections import Counter
from typing import Union

import sympy as sp

from utils.parsing import to_expr, canonicalize_expr


def _expr_tokens(expr: sp.Expr) -> list[str]:
    expr = canonicalize_expr(expr)
    s = sp.srepr(expr)
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|[^\s]", s)


def bleu_score(
    hyp: Union[str, sp.Expr],
    ref: Union[str, sp.Expr],
    max_n: int = 4,
) -> float:
    hyp_tokens = _expr_tokens(to_expr(hyp))
    ref_tokens = _expr_tokens(to_expr(ref))

    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))

        overlap = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())

        precisions.append((overlap + 1) / (total + 1))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    bp = (
        math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        if len(hyp_tokens) < len(ref_tokens)
        else 1.0
    )

    return bp * geo_mean