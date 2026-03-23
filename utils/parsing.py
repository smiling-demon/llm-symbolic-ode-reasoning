from __future__ import annotations

import re
from typing import Union

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

try:
    from sympy.parsing.latex import parse_latex
except Exception:
    parse_latex = None


_LATEX_ALIASES = {
    r"\arctg": "atan",
    r"\arctan": "atan",
    r"\ctg": "cot",
    r"\tg": "tan",
    r"\ln": "log",
    r"\log": "log",
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\cot": "cot",
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
}

_CONSTANT_RE = re.compile(r"^C(?:_\{?[A-Za-z0-9]+\}?|\d+)?$")
_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
    implicit_application,
)


def extract_boxed(text: str) -> str:
    matches = list(re.finditer(r"\\boxed\{(.*?)\}", text, re.DOTALL))
    if not matches:
        return text.strip()

    last_match = matches[-1]
    return last_match.group(1).strip()


def normalize_output(text: str) -> str:
    text = extract_boxed(text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _strip_outer_wrappers(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\operatorname\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"([A-Za-z][A-Za-z0-9_]*)\{\s*\((.*?)\)\s*\}", r"\1(\2)", s)
    return s.strip()


def _split_top_level_equal(s: str) -> tuple[str, str] | None:
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
        elif ch == "=" and depth == 0:
            return s[:i], s[i + 1 :]
    return None


def _is_strip_candidate_lhs(lhs: sp.Basic) -> bool:
    return isinstance(lhs, (sp.Symbol, AppliedUndef))


def _replace_simple_frac(s: str) -> str:
    pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    prev = None
    while prev != s:
        prev = s
        s = pattern.sub(r"(\1)/(\2)", s)
    return s


def _fallback_normalize(s: str) -> str:
    s = _strip_outer_wrappers(s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    for cmd, repl in _LATEX_ALIASES.items():
        s = re.sub(re.escape(cmd) + r"(?![A-Za-z])", repl, s)
    s = _replace_simple_frac(s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("^", "**")
    return s


def _parse_raw_latex(s: str) -> sp.Basic:
    if parse_latex is None:
        raise ValueError("parse_latex is unavailable")
    return parse_latex(s)


import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def latex_to_sympy_expr(latex: str) -> sp.Expr | None:
    raw = _strip_outer_wrappers(latex)

    try:
        parsed = _parse_raw_latex(raw)
        if isinstance(parsed, sp.Equality):
            if _is_strip_candidate_lhs(parsed.lhs):
                return sp.sympify(parsed.rhs)
            return sp.sympify(parsed.lhs - parsed.rhs)
        return sp.sympify(parsed)
    except Exception:
        pass

    try:
        norm = _fallback_normalize(raw)
        eq = _split_top_level_equal(norm)

        if eq is not None:
            lhs, rhs = eq
            lhs_expr = parse_expr(lhs, transformations=_TRANSFORMATIONS, evaluate=False)
            rhs_expr = parse_expr(rhs, transformations=_TRANSFORMATIONS, evaluate=False)
            if _is_strip_candidate_lhs(lhs_expr):
                return sp.sympify(rhs_expr)
            return sp.sympify(lhs_expr - rhs_expr)

        return sp.sympify(parse_expr(norm, transformations=_TRANSFORMATIONS, evaluate=False))
    except Exception:
        return None

def to_expr(obj: Union[str, sp.Expr]) -> sp.Expr:
    if isinstance(obj, sp.Basic):
        return sp.sympify(obj)
    return latex_to_sympy_expr(str(obj))


def canonicalize_constants(expr: sp.Expr) -> sp.Expr:
    expr = sp.sympify(expr)
    constants = sorted(
        [s for s in expr.free_symbols if _CONSTANT_RE.match(s.name)],
        key=lambda s: s.name,
    )
    mapping = {s: sp.Symbol(f"C{i+1}") for i, s in enumerate(constants)}
    return expr.xreplace(mapping)


def canonicalize_expr(expr: sp.Expr) -> sp.Expr:
    expr = canonicalize_constants(sp.sympify(expr))

    if expr.is_Atom:
        return expr

    args = [canonicalize_expr(arg) for arg in expr.args]

    if expr.is_Add or expr.is_Mul:
        args = sorted(args, key=lambda x: sp.srepr(x))
        return expr.func(*args, evaluate=False)

    if expr.is_Pow and len(args) == 2:
        return sp.Pow(args[0], args[1], evaluate=False)

    return expr.func(*args)


def strip_latex_and_normalize(expr: Union[str, sp.Expr]) -> str:
    if isinstance(expr, sp.Basic):
        text = sp.latex(expr)
    else:
        text = str(expr)
    return normalize_output(text)
