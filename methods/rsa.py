from __future__ import annotations

import random
from typing import List

from methods.base import BaseMethod


COT_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Solve the following differential equation step by step.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
1. Provide a clear step-by-step solution.
2. Be concise and don't write obvious transformations.
3. The final solution MUST be written in LaTeX.
4. The FINAL answer MUST be in the format:

Final answer: \\boxed{{<solution>}}

5. Do NOT include anything after the boxed expression.

BEGIN SOLUTION:
"""


AGGREGATION_PROMPT_TEMPLATE = """You are a mathematical expert in differential equations.

You are given a differential equation and several candidate solutions.

Problem (in LaTeX):
{question}

Candidate solutions:from llm.wrapper import LLM
{candidates}

INSTRUCTIONS:
1. Analyze all candidate solutions carefully.
2. Keep only correct and useful reasoning.
3. Discard incorrect or irrelevant steps.
4. If all candidates are weak, solve from scratch.
5. Produce one coherent step-by-step solution.
6. The final solution MUST be written in LaTeX.
7. The FINAL answer MUST be in the format:

Final answer: \\boxed{{<solution>}}

Do NOT include anything after the boxed expression.

BEGIN AGGREGATED SOLUTION:
"""


class RSA(BaseMethod):
    def __init__(
        self,
        llm,
        N: int = 3,
        K: int = 2,
        T: int = 2,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.llm = llm
        self.N = N
        self.K = K
        self.T = T
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def solve(self, equations: List[str]) -> List[str]:
        if not equations:
            return []

        populations = [[] for _ in range(len(equations))]

        prompts = []
        q_indices = []
        for q_idx, q in enumerate(equations):
            for _ in range(self.N):
                prompts.append(COT_PROMPT_TEMPLATE.format(question=q))
                q_indices.append(q_idx)

        outputs = self.llm.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )

        for out, idx in zip(outputs, q_indices):
            populations[idx].append(out)

        for _ in range(self.T):
            agg_prompts = []
            agg_indices = []

            for i, q in enumerate(equations):
                if len(populations[i]) < self.K:
                    continue

                chosen = random.sample(populations[i], self.K)
                candidates = "\n\n".join(
                    [f"CANDIDATE #{j + 1}:\n{c}" for j, c in enumerate(chosen)]
                )

                agg_prompts.append(
                    AGGREGATION_PROMPT_TEMPLATE.format(
                        question=q,
                        candidates=candidates,
                    )
                )
                agg_indices.append(i)

            if not agg_prompts:
                break

            outputs = self.llm.generate(
                agg_prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )

            new_populations = [[] for _ in range(len(equations))]
            for out, idx in zip(outputs, agg_indices):
                new_populations[idx].append(out)

            populations = new_populations

        final_outputs = []
        for pop in populations:
            final_outputs.append(pop[0] if pop else "")
        return final_outputs


def rsa(
    llm,
    equations: List[str],
    N: int = 3,
    K: int = 2,
    T: int = 1,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    return RSA(
        llm,
        N=N,
        K=K,
        T=T,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    ).solve(equations)
