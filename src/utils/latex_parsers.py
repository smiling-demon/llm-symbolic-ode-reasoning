from __future__ import annotations

import re
from typing import Optional

import sympy as sp
from sympy.parsing.latex import parse_latex as sympy_parse_latex


_BOXED_START = r"\boxed"
_CONSTANT_RE = re.compile(r"^C(?:_\{?\d+\}?|\d+)$")


def extract_boxed(text: str) -> Optional[str]:
    """
    Extracts the last \\boxed{...} expression from LaTeX text.

    The function correctly handles nested braces and returns the
    innermost balanced content of the last boxed expression.

    Args:
        text (str): Input LaTeX string.

    Returns:
        Optional[str]: Content inside the last \\boxed{...}, or None
        if no valid boxed expression exists.
    """
    if not text:
        return None

    start_token = r"\boxed{"
    last_start = text.rfind(start_token)

    if last_start == -1:
        return None

    i = last_start + len(start_token)
    brace_level = 1
    start_content = i

    while i < len(text):
        char = text[i]

        if char == "{":
            brace_level += 1
        elif char == "}":
            brace_level -= 1
            if brace_level == 0:
                return text[start_content:i].strip()

        i += 1

    return None
