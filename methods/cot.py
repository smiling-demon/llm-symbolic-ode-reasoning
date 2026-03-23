from __future__ import annotations

from typing import List

from methods.base import BaseMethod


COT_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Solve the following differential equation with several steps and transitions.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
1. Solve step by step as function y(x).
2. The final solution MUST be in LaTeX.
3. Present the FINAL answer ONLY in the format:

Final answer: \\boxed{{<solution>}}

4. Do NOT write anything after the boxed expression.

BEGIN SOLUTION:
"""


class CoT(BaseMethod):
    def __init__(self, llm, max_new_tokens: int = 1024):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def solve(self, equations: List[str]) -> List[str]:
        prompts = [COT_PROMPT_TEMPLATE.format(question=eq) for eq in equations]
        return self.llm.generate(prompts, max_new_tokens=self.max_new_tokens)


def cot(llm, equations: List[str], max_new_tokens: int = 1024) -> List[str]:
    return CoT(llm, max_new_tokens=max_new_tokens).solve(equations)
