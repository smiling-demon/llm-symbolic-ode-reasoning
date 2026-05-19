from typing import Dict, Optional, Callable
import signal
from itertools import permutations

from ..utils.sympy_parsers import ast_size, normalize, extract_constants


class TimeoutError(Exception):
    """
    Raised when execution exceeds the allowed time limit.
    """
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError()


def run_with_timeout(func: Callable, timeout: float = 1.0, fallback=None):
    """
    Executes a function with a time limit.

    Args:
        func (Callable): Function to execute.
        timeout (float): Maximum allowed execution time in seconds.
        fallback: Value returned if timeout occurs.

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


def difference_error(solution1, solution2, timeout: float = 5.0) -> float:
    """
    Computes structural difference between two symbolic solutions.

    The metric:
    - aligns constants via permutations
    - evaluates AST-based structural distance
    - normalizes by expression size

    Args:
        solution1: Reference SymPy expression.
        solution2: Predicted SymPy expression.
        timeout (float): Execution timeout in seconds.

    Returns:
        float: Normalized difference score in [0, 1],
               where 0 means identical and 1 means maximally different.
    """
    if not solution1 or not solution2:
        return 1.0

    state: Dict[str, Optional[float]] = {"best_score": None}

    def work():
        constants1 = extract_constants(solution1)
        constants2 = extract_constants(solution2)

        if len(constants1) != len(constants2):
            state["best_score"] = None
            return None

        base_expr = normalize(solution1 - solution2)
        best_score = ast_size(base_expr)

        if base_expr == 0:
            state["best_score"] = 0.0
            return 0.0

        for perm in permutations(constants2):

            if all(c1 == c2 for c1, c2 in zip(constants1, perm)):
                continue

            subs = {c2: c1 for c1, c2 in zip(constants1, perm)}

            transformed = solution2.subs(subs, simultaneous=True)
            diff = normalize(solution1 - transformed)
            score = ast_size(diff)

            if score < best_score:
                best_score = score

            if diff == 0:
                state["best_score"] = 0.0
                return 0.0

        state["best_score"] = best_score
        return min(best_score / (ast_size(solution1) + ast_size(solution2)), 1.0)

    result = run_with_timeout(work, timeout=timeout, fallback=None)

    if result is not None:
        return result

    if state["best_score"] is not None:
        return min(
            state["best_score"]
            / (ast_size(solution1) + ast_size(solution2)),
            1.0,
        )

    return 1.0
