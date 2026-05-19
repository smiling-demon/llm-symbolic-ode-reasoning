from typing import List, Optional, Dict

from ..methods import RecursiveSelfAggregation, TreeOfThought


class RSAWithToT(RecursiveSelfAggregation, TreeOfThought):
    """
    Recursive Self-Aggregation combined with Tree-of-Thought.

    Uses Tree-of-Thought reasoning to generate the initial population
    before applying recursive aggregation.
    """

    def __init__(
        self,
        llm,
        N: int = 4,
        K: int = 2,
        T: int = 2,
        max_depth: int = 4,
        beam_width: int = 2,
        thoughts_per_node: int = 2,
        vote_rounds: int = 3,
        max_state_chars: int = 2048,
        max_new_tokens: int = 1024,
        seed: Optional[int] = None,
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

        TreeOfThought.__init__(
            self,
            llm,
            max_depth,
            beam_width,
            thoughts_per_node,
            vote_rounds,
            max_state_chars,
            max_new_tokens,
            seed,
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
        Generates initial RSA population using Tree-of-Thought solver.
        """
        self._log("=== RSA+ToT INIT POPULATION START ===")

        populations: List[List[str]] = [[] for _ in questions]

        expanded_questions: List[str] = []
        q_indices: List[int] = []

        for i, q in enumerate(questions):
            for _ in range(self.N):
                expanded_questions.append(q)
                q_indices.append(i)

        self._log(f"Calling TreeOfThought.solve for {len(expanded_questions)} expansions")

        outputs = TreeOfThought.solve(self, expanded_questions, batch_size)

        self._log("TreeOfThought.solve completed")

        self._log("\n--- INITIAL POPULATION OUTPUTS ---")

        for i, out in enumerate(outputs):
            idx = q_indices[i]

            token_counts[idx] += out["token_count"]
            time_costs[idx] += out["avg_time"]

            response = out["response"]

            populations[idx].append(response)
            self._log(f"[Q{idx}] OUTPUT:\n{response}\n")

        self._log("=== RSA+ToT INIT POPULATION END ===")

        return populations
