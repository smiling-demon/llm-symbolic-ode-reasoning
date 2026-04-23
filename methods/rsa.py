from __future__ import annotations

import random
from typing import List

from methods.base import BaseMethod
from utils.parsing import extract_boxed


COT_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Solve the following differential equation step by step.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
1. Provide a clear step-by-step solution.
2. The final solution MUST be written in LaTeX.
3. The FINAL answer MUST be in the format:

Final answer: \\boxed{{<solution>}}

4. Do NOT include anything after the boxed expression.

BEGIN SOLUTION:
"""


AGGREGATION_PROMPT_TEMPLATE = """You are a mathematical expert in differential equations.

You are given a differential equation and several candidate solutions.

Problem (in LaTeX):
{question}

Candidate solutions:
{candidates}

INSTRUCTIONS:
1. Analyze all candidate solutions carefully.
2. Keep only correct and useful reasoning.
3. If all candidates are incorrect, solve from scratch.
4. Produce one coherent step-by-step solution.
5. The final solution MUST be written in LaTeX.
6. The FINAL answer MUST be in the format:

Final answer: \\boxed{{<solution>}}

7. Do NOT include anything after the boxed expression.

BEGIN AGGREGATED SOLUTION:
"""


class RSA(BaseMethod):
    def __init__(
        self,
        llm,
        N: int = 3,
        K: int = 2,
        T: int = 1,
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

    @staticmethod
    def _has_boxed_answer(text: str) -> bool:
        try:
            return extract_boxed(text) is not None
        except Exception:
            return False

    def _pick_final_answer(self, candidates: List[str]) -> str:
        valid = [c for c in candidates if self._has_boxed_answer(c)]
        if valid:
            return random.choice(valid)
        return ""

    def solve(self, equations: List[str]) -> List[str]:
        if not equations:
            return []

        populations: List[List[str]] = [[] for _ in range(len(equations))]

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
                if not populations[i]:
                    continue

                for _ in range(self.N):
                    if len(populations[i]) >= self.K:
                        chosen = random.sample(populations[i], self.K)
                    else:
                        chosen = random.choices(populations[i], k=self.K)

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

            populations = []
            for old_pop, new_pop in zip(populations if populations else [[] for _ in equations], new_populations):
                merged = new_pop if new_pop else old_pop
                boxed = [c for c in merged if self._has_boxed_answer(c)]
                if boxed:
                    if len(boxed) >= self.N:
                        populations.append(random.sample(boxed, self.N))
                    else:
                        pool = boxed[:]
                        remainder = [c for c in merged if c not in pool]
                        pool.extend(remainder)
                        while pool and len(pool) < self.N:
                            pool.append(random.choice(pool))
                        populations.append(pool[:self.N])
                else:
                    if len(merged) >= self.N:
                        populations.append(random.sample(merged, self.N))
                    else:
                        pool = merged[:]
                        while pool and len(pool) < self.N:
                            pool.append(random.choice(pool))
                        populations.append(pool[:self.N] if pool else [])

        final_outputs = []
        for pop in populations:
            final_outputs.append(self._pick_final_answer(pop))
        return final_outputs


def rsa(
    llm,
    equations: List[str],
    N: int = 3,
    K: int = 2,
    T: int = 2,
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
