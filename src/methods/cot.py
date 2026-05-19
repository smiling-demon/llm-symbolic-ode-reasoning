from typing import List

from tqdm import tqdm

from .base import BaseMethod
from ..prompts import COT_PROMPT_TEMPLATE


class ChainOfThought(BaseMethod):
    """
    Chain-of-Thought method for step-by-step reasoning generation.
    """

    def __init__(self, llm, max_new_tokens: int = 1024, logging: bool = False):
        """
        Args:
            llm: Language model with a `generate` method.
            max_new_tokens (int): Maximum number of tokens to generate.
            logging (bool): Enables logging (reserved for future use).
        """
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.logging = logging

    def solve(self, questions: List[str], batch_size: int = 4) -> List[str]:
        """
        Generates answers using a Chain-of-Thought prompting strategy.

        Args:
            questions (List[str]): Input questions.
            batch_size (int): Number of samples processed per batch.

        Returns:
            List[str]: Generated answers.
        """
        prompts = [
            COT_PROMPT_TEMPLATE.format(question=question)
            for question in questions
        ]

        outputs: List[str] = []

        for start in tqdm(range(0, len(prompts), batch_size), desc="Solve"):
            batch_prompts = prompts[start:start + batch_size]

            batch_outputs = self.llm.generate(
                batch_prompts,
                max_new_tokens=self.max_new_tokens,
            )

            outputs.extend(batch_outputs)

        return outputs
