from __future__ import annotations

import random
from typing import List, Optional, Dict

from tqdm import tqdm

from .base import BaseMethod
from ..utils.parsing import extract_boxed
from ..prompts import AGGREGATION_PROMPT_TEMPLATE, COT_PROMPT_TEMPLATE


class RecursiveSelfAggregation(BaseMethod):
    """
    Recursive Self-Aggregation (RSA) method.

    Generates multiple candidate solutions per question and iteratively
    refines them using aggregation steps.
    """

    def __init__(
        self,
        llm,
        N: int = 4,
        K: int = 2,
        T: int = 2,
        max_new_tokens: int = 1024,
        logging: Optional[str] = None,
    ):
        """
        Args:
            llm: Language model with a `generate` method.
            N (int): Number of initial candidates per question.
            K (int): Number of candidates sampled during aggregation.
            T (int): Number of refinement iterations.
            max_new_tokens (int): Max tokens per generation call.
            logging (str | None): Optional path for logging.
        """
        self.llm = llm
        self.N = N
        self.K = K
        self.T = T
        self.max_new_tokens = max_new_tokens
        self.logging = logging

    def _log(self, text: str) -> None:
        """
        Writes logs to file if logging is enabled.
        """
        if not self.logging:
            return
        with open(self.logging, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    @staticmethod
    def _has_boxed_answer(text: str) -> bool:
        """
        Checks whether a response contains a boxed answer.
        """
        try:
            return extract_boxed(text) is not None
        except Exception:
            return False

    def _pick_final_answer(self, candidates: List[str]) -> str:
        """
        Selects a final answer from candidate pool.
        """
        valid = [c for c in candidates if self._has_boxed_answer(c)]
        return random.choice(valid) if valid else ""

    # =========================================================
    # INITIAL POPULATION
    # =========================================================

    def _generate_initial_population(
        self,
        questions: List[str],
        batch_size: int,
        token_counts: List[int],
        time_costs: List[float],
    ) -> List[List[str]]:
        """
        Generates initial candidate solutions for each question.
        """
        self._log("=== INIT POPULATION START ===")

        populations: List[List[str]] = [[] for _ in questions]

        prompts: List[str] = []
        q_indices: List[int] = []

        for q_idx, q in enumerate(questions):
            for _ in range(self.N):
                prompts.append(COT_PROMPT_TEMPLATE.format(question=q))
                q_indices.append(q_idx)

        outputs: List[str] = []

        for start in tqdm(range(0, len(prompts), batch_size), desc="Init population"):
            batch_prompts = prompts[start:start + batch_size]

            batch_outputs = self.llm.generate(
                batch_prompts,
                max_new_tokens=self.max_new_tokens,
            )

            for i, o in enumerate(batch_outputs):
                q_idx = q_indices[start + i]
                token_counts[q_idx] += o["token_count"]
                time_costs[q_idx] += o["avg_time"]

            self._log(f"\n--- INIT BATCH {start} ---")
            for p, o in zip(batch_prompts, batch_outputs):
                self._log(f"PROMPT:\n{p}\nOUTPUT:\n{o['response']}\n")

            outputs.extend(o["response"] for o in batch_outputs)

        for out, idx in zip(outputs, q_indices):
            populations[idx].append(out)

        self._log("=== INIT POPULATION END ===")
        return populations

    # =========================================================
    # MAIN SOLVE
    # =========================================================

    def solve(self, questions: List[str], batch_size: int = 4) -> List[Dict]:
        """
        Solves questions using recursive self-aggregation.

        Args:
            questions (List[str]): Input questions.
            batch_size (int): Batch size for inference.

        Returns:
            List[Dict]: Structured results with response and metrics.
        """
        if not questions:
            return []

        self._log("=== RSA SOLVE START ===")

        n = len(questions)

        token_counts = [0] * n
        time_costs = [0.0] * n

        populations = self._generate_initial_population(
            questions,
            batch_size,
            token_counts,
            time_costs,
        )

        for t in range(self.T):
            self._log(f"=== ITERATION {t} START ===")

            agg_prompts: List[str] = []
            agg_indices: List[int] = []

            for i, q in enumerate(questions):
                for _ in range(self.N):
                    chosen = random.sample(
                        populations[i],
                        k=min(self.K, len(populations[i])),
                    )

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
                    agg_indices.append(i)

            outputs: List[str] = []

            for start in tqdm(
                range(0, len(agg_prompts), batch_size),
                desc=f"Aggregation {t + 1}/{self.T}",
            ):
                batch_prompts = agg_prompts[start:start + batch_size]

                batch_outputs = self.llm.generate(
                    batch_prompts,
                    max_new_tokens=self.max_new_tokens,
                )

                for i, o in enumerate(batch_outputs):
                    q_idx = agg_indices[start + i]
                    token_counts[q_idx] += o["token_count"]
                    time_costs[q_idx] += o["avg_time"]

                self._log(f"\n--- AGG BATCH {start} (t={t}) ---")
                for p, o in zip(batch_prompts, batch_outputs):
                    self._log(f"PROMPT:\n{p}\nOUTPUT:\n{o['response']}\n")

                outputs.extend(o["response"] for o in batch_outputs)

            populations = [[] for _ in questions]

            for out, idx in zip(outputs, agg_indices):
                populations[idx].append(out)

            self._log(f"=== ITERATION {t} END ===")

        # =========================================================
        # FINAL SELECTION
        # =========================================================

        final_outputs = [
            self._pick_final_answer(pop)
            for pop in populations
        ]

        self._log("=== FINAL SELECTION START ===")

        for i, pop in enumerate(populations):
            self._log(f"\n--- QUESTION {i} CANDIDATES ---")
            for j, c in enumerate(pop):
                self._log(f"CANDIDATE #{j + 1}:\n{c}\n")

        self._log("=== FINAL SELECTION END ===")
        self._log("=== RSA SOLVE END ===")

        return [
            {
                "response": final_outputs[i],
                "token_count": token_counts[i],
                "avg_time": time_costs[i],
            }
            for i in range(n)
        ]
