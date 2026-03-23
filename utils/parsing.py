from __future__ import annotations

import re
from typing import Optional, Union
import sympy as sp
from sympy.parsing.latex import parse_latex as sympy_parse_latex  # требуется antlr4


_BOXED_START = r"\boxed"
_CONSTANT_RE = re.compile(r"^C(?:_\{?\d+\}?|\d+)$")


def _read_braced_content(text: str, open_brace_idx: int) -> tuple[Optional[str], int]:
    if open_brace_idx < 0 or open_brace_idx >= len(text) or text[open_brace_idx] != "{":
        return None, open_brace_idx

    depth = 0
    content_start: Optional[int] = None
    i = open_brace_idx

    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
            if depth == 1:
                content_start = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                if content_start is None:
                    return None, open_brace_idx
                return text[content_start:i], i + 1
        i += 1
    return None, open_brace_idx


def extract_boxed(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    last_content: Optional[str] = None
    pos = 0

    while True:
        start = text.find(_BOXED_START, pos)
        if start == -1:
            break
        i = start + len(_BOXED_START)
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            pos = start + 1
            continue
        content, end_idx = _read_braced_content(text, i)
        if content is not None:
            last_content = content
            pos = end_idx
        else:
            pos = start + 1
    return last_content


def _cleanup_latex(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == "$" and text[-1] == "$":
        text = text[1:-1].strip()
    for cmd in (r"\left", r"\right", r"\displaystyle", r"\,", r"\!", r"\;", r"\:", r"\ "):
        text = text.replace(cmd, "")
    text = text.replace("−", "-")
    return text.strip()


def _extract_rhs(text: str) -> str:
    """
    Если строка вида y = ... или y(x) = ..., оставляем только правую часть.
    """
    match = re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\([^\)]*\))?\s*=\s*(.*)$", text)
    if match:
        return match.group(1).strip()
    return text


def _parse_latex(text: str) -> sp.Basic:
    text = _cleanup_latex(text)
    text = _extract_rhs(text)
    if not text:
        raise ValueError("Empty expression after cleanup.")
    try:
        return sympy_parse_latex(text)
    except Exception as exc:
        raise ValueError(f"Could not parse LaTeX: {text!r}") from exc


def to_expr(expr: Union[str, sp.Basic, sp.MatrixBase, int, float]) -> Union[sp.Basic, sp.MatrixBase]:
    if isinstance(expr, (sp.Basic, sp.MatrixBase)):
        return expr
    if isinstance(expr, (int, float)):
        return sp.sympify(expr)
    if not isinstance(expr, str):
        return sp.sympify(expr)

    text = expr.strip()
    if not text:
        raise ValueError("Empty input string.")

    return _parse_latex(text)


def _is_constant_symbol(sym: sp.Basic) -> bool:
    return isinstance(sym, sp.Symbol) and bool(_CONSTANT_RE.match(sym.name))


def _normalize_constants_for_key(expr: sp.Basic) -> sp.Basic:
    if not isinstance(expr, sp.Basic):
        return sp.sympify(expr)
    repl = {s: sp.Symbol("C") for s in expr.free_symbols if _is_constant_symbol(s)}
    return expr.xreplace(repl) if repl else expr


def _sort_key(expr: sp.Basic):
    return sp.default_sort_key(_normalize_constants_for_key(expr))


def canonicalize_expr(expr: Union[str, sp.Basic, sp.MatrixBase, int, float]) -> Union[sp.Basic, sp.MatrixBase]:
    e = to_expr(expr)
    if isinstance(e, sp.MatrixBase):
        return e.applyfunc(canonicalize_expr)

    def _canon(node: sp.Basic) -> sp.Basic:
        if isinstance(node, sp.Symbol) and _is_constant_symbol(node):
            return sp.Symbol("C")
        if node.is_Atom:
            return node
        args = tuple(_canon(arg) for arg in node.args)
        if isinstance(node, sp.Equality):
            lhs, rhs = args
            if _sort_key(rhs) < _sort_key(lhs):
                lhs, rhs = rhs, lhs
            return sp.Eq(lhs, rhs, evaluate=False)
        if node.is_Add:
            args = tuple(sorted(args, key=_sort_key))
            return sp.Add(*args, evaluate=False)
        if node.is_Mul:
            args = tuple(sorted(args, key=_sort_key))
            return sp.Mul(*args, evaluate=False)
        if node.is_Pow:
            return sp.Pow(*args, evaluate=False)
        try:
            return node.func(*args)
        except Exception:
            try:
                return node.xreplace(dict(zip(node.args, args)))
            except Exception:
                return node
    return _canon(e)
