from typing import List, Optional, Dict

from ..models import LLM, EmbeddingModel
from ..methods import RecursiveSelfAggregation, ReasoningBank


class RSAWithReasoningBank(RecursiveSelfAggregation, ReasoningBank):
    """
    Recursive Self-Aggregation combined with a Reasoning Bank.

    Uses the ReasoningBank as a structured solver during initial population
    generation instead of direct CoT generation.
    """

    def __init__(
        self,
        llm: LLM,
        embed_model: EmbeddingModel,
        N: int = 4,
        K: int = 2,
        T: int = 2,
        storage_path: str = "data/reasoning_bank.json",
        load: bool = True,
        top_k: int = 2,
        max_new_tokens: int = 1024,
        logging: Optional[str] = None,
    ):
        RecursiveSelfAggregation.__init__(
            self,
            llm,
            N,
            K,
            T,
            max_new_tokens,
            logging=logging,
        )

        ReasoningBank.__init__(
            self,
            llm,
            embed_model,
            storage_path,
            load,
            top_k,
            max_new_tokens,
            logging=logging,
        )

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
        Generates initial RSA population using ReasoningBank solver.
        """
        self._log("=== RSA+RB INIT POPULATION START ===")

        populations: List[List[str]] = [[] for _ in questions]

        expanded_questions: List[str] = []
        q_indices: List[int] = []

        for i, q in enumerate(questions):
            for _ in range(self.N):
                expanded_questions.append(q)
                q_indices.append(i)

        self._log(f"Calling ReasoningBank.solve for {len(expanded_questions)} samples")

        outputs = ReasoningBank.solve(self, expanded_questions, batch_size)

        self._log("ReasoningBank.solve completed")

        self._log("\n--- INITIAL POPULATION OUTPUTS ---")

        for i, out in enumerate(outputs):
            idx = q_indices[i]

            token_counts[idx] += out["token_count"]
            time_costs[idx] += out["avg_time"]

            response = out["response"]

            populations[idx].append(response)
            self._log(f"[Q{idx}] OUTPUT:\n{response}\n")

        self._log("=== RSA+RB INIT POPULATION END ===")

        return populations
