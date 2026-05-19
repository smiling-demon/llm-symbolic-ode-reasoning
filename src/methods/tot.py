import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tqdm import tqdm

from .base import BaseMethod
from ..utils.parsing import extract_boxed
from ..prompts import (
    TOT_GENERATE_PROMPT,
    TOT_FINAL_PROMPT,
    TOT_RANK_PROMPT,
)


# =========================================================
# REGEX
# =========================================================

RANK_RE = re.compile(r"\d+")


# =========================================================
# NODE
# =========================================================

@dataclass
class ThoughtNode:
    """
    Represents a node in the Tree-of-Thought search tree.
    """

    text: str
    thoughts: List[str]
    score: float = 0.0
    terminal: bool = False


# =========================================================
# TREE OF THOUGHT
# =========================================================

class TreeOfThought(BaseMethod):
    """
    Tree-of-Thought reasoning method.

    Builds a search tree over reasoning steps and uses
    expansion + voting to select the best path.
    """

    def __init__(
        self,
        llm,
        max_depth: int = 4,
        beam_width: int = 2,
        thoughts_per_node: int = 2,
        vote_rounds: int = 3,
        max_state_chars: int = 2048,
        max_new_tokens: int = 1024,
        seed: Optional[int] = None,
        logging: Optional[str] = None,
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.thoughts_per_node = thoughts_per_node
        self.vote_rounds = vote_rounds
        self.max_state_chars = max_state_chars
        self.max_new_tokens = max_new_tokens
        self.rng = random.Random(seed)
        self.logging = logging

    # =========================================================
    # LOGGER
    # =========================================================

    def _log(self, text: str) -> None:
        """
        Writes logs to file if enabled.
        """
        if not self.logging:
            return
        with open(self.logging, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    # =========================================================
    # PUBLIC API
    # =========================================================

    def solve(self, questions: List[str], batch_size: int = 4) -> List[dict]:
        """
        Solves questions using Tree-of-Thought search.

        Args:
            questions (List[str]): Input questions.
            batch_size (int): Batch size for LLM calls.

        Returns:
            List[dict]: Structured outputs with responses and metrics.
        """
        if not questions:
            return []

        n = len(questions)

        self._log("=== ToT SOLVE START ===")

        token_counts = [0] * n
        time_costs = [0.0] * n

        frontiers: List[List[ThoughtNode]] = [
            [ThoughtNode(text="", thoughts=[])]
            for _ in questions
        ]

        solved = [False] * n
        answers = [""] * n

        # =====================================================
        # SEARCH LOOP
        # =====================================================

        for depth in range(self.max_depth):
            active_ids = [i for i in range(n) if not solved[i]]

            if not active_ids:
                break

            self._log(f"=== DEPTH {depth} START ===")

            active_questions = [questions[i] for i in active_ids]
            active_frontiers = [frontiers[i] for i in active_ids]

            expanded = self._expand_batch(
                active_questions,
                active_frontiers,
                batch_size,
                token_counts,
                time_costs,
                active_ids,
            )

            self._vote_batch(
                active_questions,
                expanded,
                batch_size,
                token_counts,
                time_costs,
                active_ids,
            )

            for local_i, q_idx in enumerate(active_ids):
                candidates = expanded[local_i]

                candidates.sort(
                    key=lambda x: (x.score, x.terminal, -len(x.text)),
                    reverse=True,
                )

                frontiers[q_idx] = candidates[: self.beam_width]

                terminal = [c for c in candidates if c.terminal]
                if terminal:
                    best = max(terminal, key=lambda x: x.score)
                    answers[q_idx] = best.text
                    solved[q_idx] = True

            self._log(f"=== DEPTH {depth} END ===")

        # =====================================================
        # FINALIZATION
        # =====================================================

        unresolved = [i for i in range(n) if not solved[i]]

        if unresolved:
            self._log("=== FINALIZATION START ===")

            prompts = [
                TOT_FINAL_PROMPT.format(
                    question=questions[i],
                    state=self._compact(frontiers[i][0].text),
                )
                for i in unresolved
            ]

            outputs = []

            for start in tqdm(range(0, len(prompts), batch_size), desc="Finalize"):
                batch = prompts[start:start + batch_size]

                batch_out = self.llm.generate(
                    batch,
                    max_new_tokens=self.max_new_tokens,
                )

                for i, o in enumerate(batch_out):
                    q_idx = unresolved[start + i]
                    token_counts[q_idx] += o["token_count"]
                    time_costs[q_idx] += o["avg_time"]

                outputs.extend(o["response"] for o in batch_out)

            for idx, out in zip(unresolved, outputs):
                answers[idx] = frontiers[idx][0].text + "\n\n" + out

            self._log("=== FINALIZATION END ===")

        # =====================================================
        # FINAL LOG
        # =====================================================

        self._log("=== FINAL THOUGHT STRUCTURE ===")

        for q_idx, frontier in enumerate(frontiers):
            self._log(f"--- QUESTION {q_idx} ---")

            for c_idx, node in enumerate(frontier):
                self._log(
                    f"[CANDIDATE {c_idx}] SCORE={node.score} TERMINAL={node.terminal}"
                )

                for t_idx, t in enumerate(node.thoughts):
                    self._log(f"  ({t_idx}) {t}")

        self._log("=== ToT SOLVE END ===")

        return [
            {
                "response": answers[i],
                "token_count": token_counts[i],
                "avg_time": time_costs[i],
            }
            for i in range(n)
        ]

    # =========================================================
    # EXPANSION
    # =========================================================

    def _expand_batch(
        self,
        questions: List[str],
        frontiers: List[List[ThoughtNode]],
        batch_size: int,
        token_counts: List[int],
        time_costs: List[float],
        global_ids: List[int],
    ) -> List[List[ThoughtNode]]:
        """
        Expands current frontier nodes into new thought candidates.
        """
        next_frontiers: List[List[ThoughtNode]] = [[] for _ in questions]

        prompts: List[str] = []
        meta: List[Tuple[int, int]] = []

        for q_i, (q, frontier) in enumerate(zip(questions, frontiers)):
            for n_i, node in enumerate(frontier):
                state = self._compact(node.text) if node.text else "(empty)"

                for _ in range(self.thoughts_per_node):
                    prompts.append(
                        TOT_GENERATE_PROMPT.format(
                            question=q,
                            state=state,
                        )
                    )
                    meta.append((q_i, n_i))

        outputs = []

        for start in tqdm(range(0, len(prompts), batch_size), desc="Expansion"):
            batch = prompts[start:start + batch_size]

            batch_out = self.llm.generate(
                batch,
                max_new_tokens=self.max_new_tokens,
            )

            for i, o in enumerate(batch_out):
                q_i, _ = meta[start + i]
                real_q = global_ids[q_i]

                token_counts[real_q] += o["token_count"]
                time_costs[real_q] += o["avg_time"]

            outputs.extend(o["response"] for o in batch_out)

        for text, (q_i, n_i) in zip(outputs, meta):
            parent = frontiers[q_i][n_i]
            cleaned = text.strip()

            next_frontiers[q_i].append(
                ThoughtNode(
                    text=(parent.text + "\n\n" + cleaned) if parent.text else cleaned,
                    thoughts=parent.thoughts + [cleaned],
                    terminal=self._is_terminal(cleaned),
                )
            )

        return next_frontiers

    # =========================================================
    # VOTING
    # =========================================================

    def _vote_batch(
        self,
        questions: List[str],
        candidates_batch: List[List[ThoughtNode]],
        batch_size: int,
        token_counts: List[int],
        time_costs: List[float],
        global_ids: List[int],
    ) -> None:
        """
        Scores candidates using LLM-based ranking (Borda-style voting).
        """
        prompts: List[str] = []
        meta: List[Tuple[int, List[int]]] = []

        for q_i, (q, candidates) in enumerate(zip(questions, candidates_batch)):
            n = len(candidates)

            if n <= self.beam_width:
                continue

            for _ in range(self.vote_rounds):
                order = list(range(n))
                self.rng.shuffle(order)

                blocks = [
                    f"[{i}]\n```text\n{self._compact(candidates[idx].text)}\n```"
                    for i, idx in enumerate(order, 1)
                ]

                prompts.append(
                    TOT_RANK_PROMPT.format(
                        question=q,
                        candidates="\n\n".join(blocks),
                    )
                )

                meta.append((q_i, order))

        outputs = []

        for start in tqdm(range(0, len(prompts), batch_size), desc="Voting"):
            batch = prompts[start:start + batch_size]

            batch_out = self.llm.generate(
                batch,
                max_new_tokens=128,
            )

            for i, o in enumerate(batch_out):
                q_i, _ = meta[start + i]
                real_q = global_ids[q_i]

                token_counts[real_q] += o["token_count"]
                time_costs[real_q] += o["avg_time"]

            outputs.extend(o["response"] for o in batch_out)

        scores = [
            [0.0] * len(candidates)
            for candidates in candidates_batch
        ]

        for raw, (q_i, order) in zip(outputs, meta):
            n = len(order)
            ranking = self._parse_ranking(raw, n)

            if not ranking:
                continue

            for rank, local_idx in enumerate(ranking):
                global_idx = order[local_idx]
                scores[q_i][global_idx] += (n - rank)

        for q_i, candidates in enumerate(candidates_batch):
            if len(candidates) <= self.beam_width:
                for c in candidates:
                    c.score += 1.0
                continue

            for i, node in enumerate(candidates):
                node.score = scores[q_i][i] + (0.5 if node.terminal else 0.0)

    # =========================================================
    # HELPERS
    # =========================================================

    def _compact(self, text: str) -> str:
        if len(text) <= self.max_state_chars:
            return text

        h = self.max_state_chars // 2
        return text[:h] + "\n...\n" + text[-h:]

    @staticmethod
    def _is_terminal(text: str) -> bool:
        return extract_boxed(text) is not None

    @staticmethod
    def _parse_ranking(raw: str, n: int) -> List[int]:
        nums = RANK_RE.findall(raw)

        try:
            ranking = [int(x) for x in nums]
        except ValueError:
            return []

        if len(ranking) != n:
            return []

        if set(ranking) != set(range(1, n + 1)):
            return []

        return [x - 1 for x in ranking]
