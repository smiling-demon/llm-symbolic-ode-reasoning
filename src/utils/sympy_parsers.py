import re
import latex2sympy2
from latex2sympy2 import latex2sympy
from sympy import (
    together,
    cancel,
    factor,
    powsimp,
    trigsimp,
    simplify,
    preorder_traversal,
)


# =========================================================
# BASIC UTILITIES
# =========================================================

def ast_size(expr) -> float:
    """
    Computes AST (abstract syntax tree) size of a SymPy expression.

    Args:
        expr: SymPy expression.

    Returns:
        float: Number of nodes in the expression tree.
    """
    return len(list(preorder_traversal(expr))) if expr != 0 else 0.0


def safe_latex2sympy(expr: str):
    """
    Safely converts LaTeX to SymPy expression using latex2sympy2.

    Args:
        expr (str): LaTeX expression.

    Returns:
        SymPy expression or error string.
    """
    try:
        latex2sympy2.var = {}
        return latex2sympy(expr)
    except Exception as e:
        return f"ERROR: {e}"


# =========================================================
# SANITIZATION
# =========================================================

def sanitize_latex(expr: str) -> str:
    """
    Normalizes LaTeX expressions into a form compatible with latex2sympy2.

    Args:
        expr (str): Raw LaTeX expression.

    Returns:
        str: Cleaned LaTeX string.
    """
    replacements = {
        r"\operatorname{ctan}": r"\cot",
        r"\operatorname{acos}": r"\arccos",
        r"\operatorname{acot}": r"\arccot",
        r"\operatorname{asin}": r"\arcsin",
        r"\operatorname{atan}": r"\arctan",

        r"\arcctan": r"\arccot",
        r"\arctg": r"\arctan",
        r"\arcctg": r"\arccot",
        r"\arcot": r"\arccot",

        r"\ctg": r"\cot",
        r"\ctan": r"\cot",
        r" sh": r" \sinh",
        r" ch": r" \cosh",
        r"\tg": r"\tan",

        r"xi_{1}": r"xi",
    }

    for old, new in replacements.items():
        expr = expr.replace(old, new)

    return expr


# =========================================================
# PRIME NOTATION
# =========================================================

def latex_primes_to_derivative(expr: str) -> str:
    """
    Converts prime notation into explicit derivative form.

    Example:
        y'' -> d/dx(d/dx(y(x)))
    """

    def build_derivative(n: int) -> str:
        result = "y(x)"
        for _ in range(n):
            result = f"\\frac{{d}}{{dx}}({result})"
        return result

    def repl(match):
        content = match.group(1)
        n = content.count("\\prime") + content.count("'")
        return build_derivative(n)

    expr = re.sub(r"y\^\{([^}]+)\}", repl, expr)
    expr = re.sub(r"(?<![A-Za-z])y(?!\()", "y(x)", expr)

    return expr


# =========================================================
# EVALUATION PARSING
# =========================================================

def parse_evaluation(expr: str):
    """
    Parses evaluation expressions of the form:
    \\left. ... \\right|_{\\substack{...}}

    Returns SymPy expression with substitutions applied.
    """
    start = expr.find(r"\left.")
    end = expr.find(r"\right|_")

    if start == -1 or end == -1:
        return safe_latex2sympy(expr)

    inside = expr[start + len(r"\left."):end].strip()

    sub_match = re.search(
        r"\\substack\{\s*([^=]+)=([^}]*)\s*\}",
        expr,
    )

    if not sub_match:
        raise ValueError("No substack found")

    var_latex = sub_match.group(1)
    value_latex = sub_match.group(2)

    sym_expr = safe_latex2sympy(inside)
    sym_var = safe_latex2sympy(var_latex)
    sym_val = safe_latex2sympy(value_latex)

    return simplify(sym_expr.subs(sym_var, sym_val))


def extract_evaluation_blocks(expr: str):
    """
    Extracts evaluation blocks and replaces them with placeholders.

    Returns:
        (clean_expr, dict[placeholder -> sympy expression])
    """
    placeholders = {}
    counter = 0

    while r"\left." in expr:
        start = expr.find(r"\left.")
        end = expr.find(r"\right|_", start)

        if end == -1:
            break

        brace_start = expr.find("{", end)

        depth = 0
        brace_end = None

        for i in range(brace_start, len(expr)):
            if expr[i] == "{":
                depth += 1
            elif expr[i] == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i
                    break

        if brace_end is None:
            break

        full_block = expr[start:brace_end + 1]
        parsed = parse_evaluation(full_block)

        placeholder = f"\\eta_{{{counter}}}"
        placeholders[placeholder] = parsed

        expr = expr.replace(full_block, placeholder)
        counter += 1

    return expr, placeholders


# =========================================================
# MAIN PIPELINE
# =========================================================

def latex_to_sympy(expr: str):
    """
    Converts LaTeX expression to SymPy with full preprocessing pipeline.
    """
    expr = sanitize_latex(expr)
    expr = latex_primes_to_derivative(expr)
    expr, placeholders = extract_evaluation_blocks(expr)

    parsed = safe_latex2sympy(expr)

    for key, value in placeholders.items():
        parsed = parsed.subs(safe_latex2sympy(key), value)

    return parsed


def split_outside_parens(expr: str):
    """
    Splits equation into LHS and RHS by '=' outside parentheses.
    """
    depth = 0

    for i, ch in enumerate(expr):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "=" and depth == 0:
            return expr[:i], expr[i + 1:]

    return expr, "0"


def equation_to_sympy(latex: str):
    """
    Converts LaTeX equation into SymPy (lhs, rhs).
    """
    expr = sanitize_latex(latex)
    lhs, rhs = split_outside_parens(expr)

    lhs = latex_primes_to_derivative(lhs)
    rhs = latex_primes_to_derivative(rhs)

    try:
        lhs_sympy = latex_to_sympy(lhs)
    except Exception as e:
        lhs_sympy = f"ERROR: {e}"

    try:
        rhs_sympy = latex_to_sympy(rhs)
    except Exception as e:
        rhs_sympy = f"ERROR: {e}"

    return lhs_sympy, rhs_sympy


def solution_to_sympy(latex: str):
    """
    Converts solution LaTeX into SymPy expression.
    """
    expr = sanitize_latex(latex)

    if "=" in expr:
        _, rhs = split_outside_parens(expr)
    else:
        rhs = expr

    try:
        return latex_to_sympy(rhs)
    except Exception as e:
        return f"ERROR: {e}"


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(expr):
    """
    Applies standard symbolic simplifications.
    """
    expr = together(expr)
    expr = cancel(expr)
    expr = factor(expr)
    expr = powsimp(expr)
    expr = trigsimp(expr)
    expr = simplify(expr)
    return expr


def extract_constants(expr):
    """
    Extracts integration constants C1, C2, C3, ...
    """
    return sorted(
        [s for s in expr.free_symbols if s.name.startswith("C")],
        key=lambda s: s.name,
    )
