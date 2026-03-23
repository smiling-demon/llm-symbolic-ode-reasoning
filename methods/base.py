from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseMethod(ABC):
    @abstractmethod
    def solve(self, equations: List[str]) -> List[str]:
        raise NotImplementedError
