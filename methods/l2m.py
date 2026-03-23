from __future__ import annotations

import re
from typing import List

from methods.base import BaseMethod


DECOMPOSITION_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Decompose the following differential equation into a minimal ordered list of at most 3 simpler subproblems needed to solve it.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
1. Output only the decomposition, not the solution.
2. Use no more than 3 subproblems.
3. Each subproblem must be simpler than the original problem.
4. Keep the decomposition minimal and logically ordered.
5. Write each subproblem in plain natural language (no LaTeX, no formulas, no symbolic fragments).
6. Describe steps conceptually.
7. Write each subproblem on a separate line using a numbered list.
8. Do not solve the subproblems.
9. Do not include any extra text outside the numbered list.

BEGIN DECOMPOSITION:
"""


SUBPROBLEM_SOLVING_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Original problem (in LaTeX):
{question}

Current subproblem:
{subproblem}

Previously solved subproblems:
{history}

INSTRUCTIONS:
1. Solve ONLY the current subproblem.
2. Use the previously solved subproblems as context.
3. Be concise.

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

    return subproblems[:3]


class LeastToMostBatched(BaseMethod):
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

        all_subproblems = [
            _parse_subproblems(d) for d in decompositions
        ]

        histories = [[] for _ in equations]

        max_steps = max((len(sp) for sp in all_subproblems), default=0)

        for step_idx in range(max_steps):
            batch_prompts = []
            batch_indices = []

            for i, (eq, subproblems) in enumerate(zip(equations, all_subproblems)):
                if step_idx < len(subproblems):
                    history_text = (
                        "\n".join(histories[i]) if histories[i] else "None"
                    )

                    prompt = SUBPROBLEM_SOLVING_PROMPT_TEMPLATE.format(
                        question=eq,
                        subproblem=subproblems[step_idx],
                        history=history_text,
                    )

                    batch_prompts.append(prompt)
                    batch_indices.append(i)

            if not batch_prompts:
                continue

            outputs = self.llm.generate(
                batch_prompts,
                max_new_tokens=self.max_new_tokens,
            )

            for out, i, step_i in zip(outputs, batch_indices, range(len(batch_indices))):
                subproblem_text = all_subproblems[i][step_idx]

                histories[i].append(
                    f"Step {step_idx + 1}: {subproblem_text}\nResult: {out.strip()}"
                )

        final_prompts = []

        for eq, history in zip(equations, histories):
            history_text = "\n\n".join(history) if history else "None"

            final_prompts.append(
                FINAL_SYNTHESIS_PROMPT_TEMPLATE.format(
                    question=eq,
                    history=history_text,
                )
            )

        final_outputs = self.llm.generate(
            final_prompts,
            max_new_tokens=self.max_new_tokens,
        )

        return final_outputs


def least_to_most_batched(
    llm,
    equations: List[str],
    max_new_tokens: int = 1024,
) -> List[str]:
    return LeastToMostBatched(llm, max_new_tokens=max_new_tokens).solve(equations)