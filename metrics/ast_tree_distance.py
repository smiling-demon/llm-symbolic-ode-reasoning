from __future__ import annotations

from typing import Union

import sympy as sp
from zss import Node, simple_distance

from utils.parsing import to_expr, canonicalize_expr


def sympy_to_tree(expr: sp.Expr):
    expr = canonicalize_expr(expr)

    label = str(expr) if expr.is_Atom else expr.func.__name__
    node = Node(label)

    for arg in expr.args:
        node.addkid(sympy_to_tree(arg))

    return node


def ast_tree_distance(
    expr1: Union[str, sp.Expr],
    expr2: Union[str, sp.Expr],
) -> float:
    e1 = canonicalize_expr(to_expr(expr1))
    e2 = canonicalize_expr(to_expr(expr2))

    t1 = sympy_to_tree(e1)
    t2 = sympy_to_tree(e2)

    dist = simple_distance(t1, t2)

    size = max(len(list(sp.preorder_traversal(e1))),
               len(list(sp.preorder_traversal(e2))), 1)

    return float(dist) / float(size)
