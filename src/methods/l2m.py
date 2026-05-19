import re
from typing import List, Dict

from tqdm import tqdm

from .base import BaseMethod
from ..prompts import (
    DECOMPOSITION_PROMPT_TEMPLATE,
    SUBPROBLEM_SOLVING_PROMPT_TEMPLATE,
    FINAL_SYNTHESIS_PROMPT_TEMPLATE,
)


class LeastToMost(BaseMethod):
    """
    Least-to-Most prompting strategy.

    The method:
    1. Decomposes a question into subproblems
    2. Solves subproblems sequentially while maintaining history
    3. Produces a final synthesis answer
    """

    def __init__(self, llm, max_new_tokens: int = 1024, logging: str | None = None):
        """
        Args:
            llm: Language model with a `generate` method.
            max_new_tokens (int): Maximum number of tokens to generate per call.
            logging (str | None): Optional path to a log file.
        """
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.logging = logging

    def _log(self, text: str) -> None:
        """
        Writes logs to a file if logging is enabled.

        Args:
            text (str): Text to log.
        """
        if not self.logging:
            return
        with open(self.logging, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def solve(self, questions: List[str], batch_size: int = 4) -> List[Dict]:
        """
        Solves questions using Least-to-Most decomposition.

        Args:
            questions (List[str]): Input questions.
            batch_size (int): Batch size for inference.

        Returns:
            List[Dict]: Each dict contains:
                - response (str): final answer
                - token_count (int): total generated tokens
                - avg_time (float): average generation time
        """
        if not questions:
            return []

        self._log("=== LeastToMost RUN START ===")

        results: List[Dict] = []

        question_tokens = [0] * len(questions)
        question_time = [0.0] * len(questions)

        # =====================================================
        # DECOMPOSITION
        # =====================================================

        decomposition_prompts = [
            DECOMPOSITION_PROMPT_TEMPLATE.format(question=q)
            for q in questions
        ]

        decompositions: List[str] = []

        for start in tqdm(range(0, len(decomposition_prompts), batch_size), desc="Decomposition"):
            batch_prompts = decomposition_prompts[start:start + batch_size]

            batch_outputs = self.llm.generate(
                batch_prompts,
                max_new_tokens=self.max_new_tokens,
            )

            for i, o in enumerate(batch_outputs):
                idx = start + i
                question_tokens[idx] += o["token_count"]
                question_time[idx] += o["avg_time"]

            self._log(f"\n--- Decomposition batch {start} ---")
            for p, o in zip(batch_prompts, batch_outputs):
                self._log(f"PROMPT:\n{p}\nOUTPUT:\n{o['response']}\n")

            decompositions.extend(o["response"] for o in batch_outputs)

        # =====================================================
        # SUBPROBLEMS
        # =====================================================

        all_subproblems = [self._parse_subproblems(d) for d in decompositions]
        histories: List[List[str]] = [[] for _ in questions]
        final_answers = [""] * len(questions)

        max_steps = max((len(sp) for sp in all_subproblems), default=0)

        # =====================================================
        # STEP SOLVING LOOP
        # =====================================================

        for step_idx in range(max_steps):
            step_prompts: List[str] = []
            batch_indices: List[int] = []

            for i, (question, subproblems) in enumerate(zip(questions, all_subproblems)):
                if step_idx < len(subproblems):
                    history_text = "\n".join(histories[i]) or "(empty)"

                    prompt = SUBPROBLEM_SOLVING_PROMPT_TEMPLATE.format(
                        question=question,
                        subproblem=subproblems[step_idx],
                        history=history_text,
                    )

                    step_prompts.append(prompt)
                    batch_indices.append(i)

            if not step_prompts:
                continue

            outputs_text: List[str] = []

            for start in tqdm(
                range(0, len(step_prompts), batch_size),
                desc=f"Step {step_idx + 1}"
            ):
                batch_prompts = step_prompts[start:start + batch_size]

                batch_outputs = self.llm.generate(
                    batch_prompts,
                    max_new_tokens=self.max_new_tokens,
                )

                for i, o in enumerate(batch_outputs):
                    idx = batch_indices[start + i]

                    question_tokens[idx] += o["token_count"]
                    question_time[idx] += o["avg_time"]

                    outputs_text.append(o["response"])

                self._log(f"\n--- Step {step_idx} batch {start} ---")
                for p, o in zip(batch_prompts, batch_outputs):
                    self._log(f"PROMPT:\n{p}\nOUTPUT:\n{o['response']}\n")

            for out, i in zip(outputs_text, batch_indices):
                subproblem_text = all_subproblems[i][step_idx]

                histories[i].append(
                    f"Step {step_idx + 1}: {subproblem_text}\nResult: {out.strip()}"
                )

                final_answers[i] = out.strip()

        # =====================================================
        # FINAL SYNTHESIS
        # =====================================================

        final_prompts: List[str] = []

        for question, history in zip(questions, histories):
            history_text = "\n\n".join(history) or "None"

            final_prompts.append(
                FINAL_SYNTHESIS_PROMPT_TEMPLATE.format(
                    question=question,
                    history=history_text,
                )
            )

        for start in tqdm(range(0, len(final_prompts), batch_size), desc="Final synthesis"):
            batch_prompts = final_prompts[start:start + batch_size]

            batch_outputs = self.llm.generate(
                batch_prompts,
                max_new_tokens=self.max_new_tokens,
            )

            for i, o in enumerate(batch_outputs):
                idx = start + i

                question_tokens[idx] += o["token_count"]
                question_time[idx] += o["avg_time"]

                final_answers[idx] = o["response"].strip()

            self._log(f"\n--- Final batch {start} ---")
            for p, o in zip(batch_prompts, batch_outputs):
                self._log(f"PROMPT:\n{p}\nOUTPUT:\n{o['response']}\n")

        self._log("=== LeastToMost RUN END ===")

        for i in range(len(questions)):
            results.append(
                {
                    "response": final_answers[i],
                    "token_count": question_tokens[i],
                    "avg_time": question_time[i],
                }
            )

        return results

    @staticmethod
    def _parse_subproblems(decomposition_text: str) -> List[str]:
        """
        Extracts subproblems from a decomposition text.

        Args:
            decomposition_text (str): Raw model decomposition output.

        Returns:
            List[str]: List of extracted subproblems (max 3).
        """
        subproblems: List[str] = []

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
