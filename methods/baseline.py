from __future__ import annotations

from typing import List

from methods.base import BaseMethod


BASELINE_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Solve the following differential equation.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
1. Do NOT write steps of solution, you need to write ONLY final answer.
1. The final solution MUST be in LaTeX.
2. Present the FINAL answer ONLY in the format:

Final answer: \\boxed{{<solution>}}

3. Do NOT write anything after the boxed expression.

BEGIN SOLUTION:
"""


class Baseline(BaseMethod):
    def __init__(self, llm, max_new_tokens: int = 1024):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def solve(self, equations: List[str]) -> List[str]:
        prompts = [BASELINE_PROMPT_TEMPLATE.format(question=eq) for eq in equations]
        return self.llm.generate(prompts, max_new_tokens=self.max_new_tokens)


def baseline(llm, equations: List[str], max_new_tokens: int = 1024) -> List[str]:
    return Baseline(llm, max_new_tokens=max_new_tokens).solve(equations)
