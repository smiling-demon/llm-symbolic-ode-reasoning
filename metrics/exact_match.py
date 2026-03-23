from __future__ import annotations

from typing import Union

from utils.parsing import to_expr, canonicalize_expr
import sympy as sp


def exact_match(
    expr1: Union[str, sp.Expr],
    expr2: Union[str, sp.Expr],
) -> bool:
    e1 = canonicalize_expr(to_expr(expr1))
    e2 = canonicalize_expr(to_expr(expr2))

    return True if sp.simplify(e1 - e2) == 0 else False
