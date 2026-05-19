from typing import Optional, Callable
import signal
import sympy as sp

from ..utils.sympy_parsers import ast_size, normalize


class TimeoutError(Exception):
    """
    Raised when execution exceeds the allowed time limit.
    """
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError()


def run_with_timeout(func: Callable, timeout: float = 1.0, fallback=None):
    """
    Executes a function with a time limit using SIGALRM.

    Args:
        func (Callable): Function to execute.
        timeout (float): Timeout in seconds.
        fallback: Value returned if execution times out.

    Returns:
        Any: Function result or fallback value.
    """
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout))

    try:
        return func()
    except TimeoutError:
        return fallback
    finally:
        signal.alarm(0)


def substitution_error(equation, solution, timeout: float = 5.0) -> float:
    """
    Computes residual error of a symbolic solution substituted into an equation.

    The function:
    - substitutes solution into both sides of equation
    - replaces derivatives with symbolic derivatives of the solution
    - measures structural mismatch using AST size

    Args:
        equation: Tuple (lhs, rhs) SymPy expressions.
        solution: Candidate solution expression.
        timeout (float): Maximum execution time.

    Returns:
        float: Normalized error in [0, 1], where 0 means perfect fit.
    """
    if not equation or not solution:
        return 1.0

    def work():
        lhs, rhs = equation

        x = sp.Symbol("x")
        y = sp.Function("y")

        lhs_sub = lhs.subs(y(x), solution)
        rhs_sub = rhs.subs(y(x), solution)

        residual = lhs_sub - rhs_sub

        for deriv in residual.atoms(sp.Derivative):

            if deriv.expr == y(x):
                order = deriv.variables.count(x)
                replacement = sp.diff(solution, x, order)

                lhs_sub = lhs_sub.subs(deriv, replacement)
                rhs_sub = rhs_sub.subs(deriv, replacement)

        diff = normalize(lhs_sub - rhs_sub)

        return min(
            ast_size(diff) / (ast_size(lhs) + ast_size(rhs)),
            1.0,
        )

    return run_with_timeout(work, timeout=timeout, fallback=1.0)
