from __future__ import annotations

import re
from typing import List

from methods.base import BaseMethod


DECOMPOSITION_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Decompose the following differential equation into the smallest ordered list of simpler subproblems needed to solve it.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
1. Output only the decomposition, not the solution.
2. Each subproblem must be simpler than the original problem.
3. Keep the decomposition minimal and logically ordered.
4. Write each subproblem on a separate line using a numbered list.
5. Do not solve the subproblems here.
6. Do not include any extra text outside the numbered list.

Example format:
1. ...
2. ...
3. ...

BEGIN DECOMPOSITION:
"""


SUBPROBLEM_SOLVING_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

We are using the Least-to-Most strategy.

Original problem (in LaTeX):
{question}

Current subproblem:
{subproblem}

Previously solved subproblems:
{history}

INSTRUCTIONS:
1. Solve only the current subproblem.
2. Use the previously solved subproblems as context.
3. Show clear step-by-step reasoning.
4. Be concise and don't write obvious transformations.

BEGIN SOLUTION:
"""


FINAL_SYNTHESIS_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Original problem (in LaTeX):
{question}

Solved subproblems:
{history}

INSTRUCTIONS:
1. Use the solved subproblems to solve the original problem.
2. The final solution MUST be written in LaTeX.
3. The FINAL answer MUST be in the format:

Final answer: \\boxed{{<solution>}}

4. Do not include anything after the boxed expression.

BEGIN FINAL SOLUTION:
"""


def _parse_subproblems(decomposition_text: str) -> List[str]:
    subproblems = []
    for line in decomposition_text.splitlines():
        line = line.strip()
        match = re.match(r"^(?:\d+[\.\)\-:]*|\-|\*)\s*(.+)$", line)
        if match:
            item = match.group(1).strip()
            if item:
                subproblems.append(item)

    if not subproblems:
        cleaned = decomposition_text.strip()
        if cleaned:
            subproblems = [cleaned]

    return subproblems


class LeastToMost(BaseMethod):
    def __init__(self, llm, max_new_tokens: int = 1024):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def solve(self, equations: List[str]) -> List[str]:
        if not equations:
            return []

        decomposition_prompts = [
            DECOMPOSITION_PROMPT_TEMPLATE.format(question=eq)
            for eq in equations
        ]
        decompositions = self.llm.generate(
            decomposition_prompts,
            max_new_tokens=self.max_new_tokens,
        )

        final_outputs = []

        for eq, decomposition in zip(equations, decompositions):
            print("AAA", decomposition, "\nAAA")
            subproblems = _parse_subproblems(decomposition)

            history = []
            for idx, subproblem in enumerate(subproblems, start=1):
                history_text = "\n\n".join(history) if history else "None"
                prompt = SUBPROBLEM_SOLVING_PROMPT_TEMPLATE.format(
                    question=eq,
                    subproblem=subproblem,
                    history=history_text,
                )
                step_output = self.llm.generate(
                    [prompt],
                    max_new_tokens=self.max_new_tokens,
                )[0]

                history.append(
                    f"Subproblem #{idx}:\n{subproblem}\n\nSolution:\n{step_output}"
                )

            final_prompt = FINAL_SYNTHESIS_PROMPT_TEMPLATE.format(
                question=eq,
                history="\n\n".join(history) if history else "None",
            )
            final_output = self.llm.generate(
                [final_prompt],
                max_new_tokens=self.max_new_tokens,
            )[0]

            final_outputs.append(final_output)

        return final_outputs


def least_to_most(llm, equations: List[str], max_new_tokens: int = 1024) -> List[str]:
    return LeastToMost(llm, max_new_tokens=max_new_tokens).solve(equations)
