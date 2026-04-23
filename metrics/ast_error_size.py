from __future__ import annotations

from typing import Union

import sympy as sp
from sympy import preorder_traversal

from utils.parsing import to_expr, canonicalize_expr


def ast_size(expr: Union[str, sp.Expr]) -> int:
    e = to_expr(expr)
    return sum(1 for _ in preorder_traversal(e))


def ast_error_size(
    expr1: Union[str, sp.Expr],
    expr2: Union[str, sp.Expr],
) -> float:
    e1 = canonicalize_expr(to_expr(expr1))
    e2 = canonicalize_expr(to_expr(expr2))

    diff = sp.simplify(e1 - e2)
    if diff == 0:
        return 0.0

    size_diff = ast_size(diff)
    size_ref = ast_size(e1) + ast_size(e2)

    return min(float(size_diff) / float(size_ref), 1)
