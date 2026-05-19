from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseMethod(ABC):
    """
    Abstract base class for all solving methods.

    All subclasses must implement the `solve` method,
    which takes a list of questions and returns a list of answers.
    """

    @abstractmethod
    def solve(self, questions: List[str], **kwargs) -> List[str]:
        """
        Solves a batch of questions.

        Args:
            questions (List[str]): Input questions.
            **kwargs: Additional method-specific parameters.

        Returns:
            List[str]: Generated answers.
        """
        raise NotImplementedError
