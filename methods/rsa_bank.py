from __future__ import annotations

import random
from typing import List

from methods.base import BaseMethod
from utils.parsing import extract_boxed

from .bank import (
    ReasoningBank,
    MemoryRetriever,
    _extract_core_concept,
    QUESTION_PROMPT_WITH_MEMORIES,
)

from .rsa import AGGREGATION_PROMPT_TEMPLATE


class RSABank(BaseMethod):
    def __init__(
        self,
        llm,
        bank: ReasoningBank,
        embed_model,
        N: int = 3,
        K: int = 2,
        T: int = 1,
        top_k_memories: int = 2,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.llm = llm
        self.bank = bank
        self.embed_model = embed_model

        self.N = N
        self.K = K
        self.T = T

        self.top_k_memories = top_k_memories
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p


    @staticmethod
    def _has_boxed(text: str) -> bool:
        try:
            return extract_boxed(text) is not None
        except Exception:
            return False

    def _choose_k(self, pop: List[str]) -> List[str]:
        if len(pop) >= self.K:
            return random.sample(pop, self.K)
        return random.choices(pop, k=self.K)


    def _build_initial_population(self, equations: List[str]) -> List[List[str]]:
        retriever = MemoryRetriever(self.embed_model)
        concepts = _extract_core_concept(self.llm, equations)

        memory_contexts = []
        for eq, concept in zip(equations, concepts):
            retrieved = retriever.retrieve(
                query=concept,
                memories=self.bank.get_all_memories(),
                top_k=self.top_k_memories,
            )
            memory_contexts.append(
                retriever.format_memories_for_prompt(retrieved)
            )

        prompts = []
        idx_map = []

        for i, (eq, mem) in enumerate(zip(equations, memory_contexts)):
            for _ in range(self.N):
                prompts.append(
                    QUESTION_PROMPT_WITH_MEMORIES.format(
                        question=eq,
                        memories=mem,
                    )
                )
                idx_map.append(i)

        outputs = self.llm.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )

        populations: List[List[str]] = [[] for _ in equations]

        for out, i in zip(outputs, idx_map):
            populations[i].append(out)

        return populations

    def solve(self, equations: List[str]) -> List[str]:
        if not equations:
            return []

        populations = self._build_initial_population(equations)

        for _ in range(self.T):
            agg_prompts = []
            agg_idx = []

            for i, q in enumerate(equations):
                if not populations[i]:
                    continue

                for _ in range(self.N):
                    chosen = self._choose_k(populations[i])

                    candidates = "\n\n".join(
                        f"CANDIDATE #{j + 1}:\n{c}"
                        for j, c in enumerate(chosen)
                    )

                    agg_prompts.append(
                        AGGREGATION_PROMPT_TEMPLATE.format(
                            question=q,
                            candidates=candidates,
                        )
                    )
                    agg_idx.append(i)

            if not agg_prompts:
                break

            outputs = self.llm.generate(
                agg_prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )

            new_populations = [[] for _ in equations]

            for out, i in zip(outputs, agg_idx):
                new_populations[i].append(out)

            populations = []

            for pop in new_populations:
                boxed = [x for x in pop if self._has_boxed(x)]

                if boxed:
                    if len(boxed) >= self.N:
                        populations.append(random.sample(boxed, self.N))
                    else:
                        pool = boxed[:]
                        while len(pool) < self.N:
                            pool.append(random.choice(pool))
                        populations.append(pool[:self.N])
                else:
                    if len(pop) >= self.N:
                        populations.append(random.sample(pop, self.N))
                    else:
                        pool = pop[:]
                        while len(pool) < self.N and pool:
                            pool.append(random.choice(pool))
                        populations.append(pool[:self.N] if pool else [])


        results = []
        for pop in populations:
            valid = [p for p in pop if self._has_boxed(p)]
            if valid:
                results.append(random.choice(valid))
            else:
                results.append("")

        return results


def rsa_bank(
    llm,
    bank: ReasoningBank,
    embed_model,
    equations: List[str],
    N: int = 3,
    K: int = 2,
    T: int = 2,
    top_k_memories: int = 2,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    return RSABank(
        llm=llm,
        bank=bank,
        embed_model=embed_model,
        N=N,
        K=K,
        T=T,
        top_k_memories=top_k_memories,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    ).solve(equations)
