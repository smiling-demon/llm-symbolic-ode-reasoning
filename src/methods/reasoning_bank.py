import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Dict

import numpy as np
from tqdm import tqdm

from .base import BaseMethod
from ..models import EmbeddingModel
from ..utils.parsing import extract_boxed
from ..prompts import (
    SUCCESS_PROMPT,
    FAILURE_PROMPT,
    QUESTION_PROMPT,
    QUESTION_PROMPT_WITH_MEMORIES,
    KEY_CONCEPT_EXTRACTION_PROMPT,
    JUDGE_PROMPT,
)


# =========================================================
# MEMORY
# =========================================================

@dataclass
class MemoryItem:
    """
    A single stored reasoning memory item.
    """

    title: str
    description: str
    content: str
    source_problem_id: str
    success: bool
    created_at: str
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        """Converts memory item to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryItem":
        """Creates MemoryItem from dictionary."""
        return cls(**data)


# =========================================================
# JUDGE
# =========================================================

class LLMJudge:
    """
    Uses an LLM to evaluate whether predicted answers match references.
    """

    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _parse_yes_no(text: str) -> bool:
        """
        Parses YES/NO responses robustly.
        """
        if not text:
            return False
        first = re.split(r"\s+", text.strip().upper(), maxsplit=1)[0]
        return first == "YES"

    def judge_batch(
        self,
        questions: Sequence[str],
        predicted_answers: Sequence[str],
        reference_answers: Sequence[str],
        max_new_tokens: int = 8,
    ) -> List[bool]:
        """
        Judges a batch of predictions.

        Returns:
            List[bool]: correctness flags.
        """
        if not (len(questions) == len(predicted_answers) == len(reference_answers)):
            raise ValueError("All inputs must have the same length")

        prompts = [
            JUDGE_PROMPT.format(question=q, predicted=p, reference=r)
            for q, p, r in zip(questions, predicted_answers, reference_answers)
        ]

        outputs = self.llm.generate(prompts, max_new_tokens=max_new_tokens)

        return [self._parse_yes_no(o["response"]) for o in outputs]


# =========================================================
# RETRIEVER
# =========================================================

class MemoryRetriever:
    """
    Retrieves relevant memories using embedding similarity.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.model = embedding_model

    def embed_memories(self, memories: List[MemoryItem]) -> None:
        """
        Computes and stores embeddings for memories.
        """
        texts = [f"{m.title}. {m.description}. {m.content}" for m in memories]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        for memory, emb in zip(memories, embeddings):
            memory.embedding = emb.astype(np.float32).tolist()

    def retrieve(
        self,
        query: str,
        memories: List[MemoryItem],
        top_k: int = 2,
        similarity_threshold: float = 0.95,
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieves most relevant memories for a query.
        """
        if not memories:
            return []

        if any(m.embedding is None for m in memories):
            self.embed_memories(memories)

        query_emb = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        memory_embs = np.array([m.embedding for m in memories], dtype=np.float32)
        sims = memory_embs @ query_emb

        k = min(len(memories), max(1, 2 * top_k))
        top_indices = np.argsort(sims)[-k:][::-1]

        selected: List[Tuple[MemoryItem, float]] = []
        selected_embs: List[np.ndarray] = []

        for idx in top_indices:
            emb = memory_embs[idx]
            sim = float(sims[idx])

            if not selected_embs:
                selected.append((memories[idx], sim))
                selected_embs.append(emb)
            else:
                max_sim = max(float(np.dot(emb, e)) for e in selected_embs)
                if max_sim < similarity_threshold:
                    selected.append((memories[idx], sim))
                    selected_embs.append(emb)

            if len(selected) >= top_k:
                break

        return selected

    def format_memories_for_prompt(
        self,
        retrieved: List[Tuple[MemoryItem, float]],
    ) -> str:
        """
        Formats retrieved memories into a prompt string.
        """
        if not retrieved:
            return ""

        text = "Past Strategy Hints:\n\n"
        text += "IMPORTANT: Do NOT copy numbers from these hints.\n\n"

        for i, (m, _) in enumerate(retrieved, 1):
            status = "Success Strategy" if m.success else "Failure Lesson"
            text += f"## Strategy {i} ({status}):\n"
            text += f"Title: {m.title}\n"
            text += f"{m.content}\n\n"

        return text


# =========================================================
# EXTRACTOR
# =========================================================

class MemoryExtractor:
    """
    Extracts structured memories from reasoning trajectories.
    """

    def __init__(self, llm):
        self.llm = llm

    def extract_from_trajectories(self, trajectories: List[Dict]) -> List[MemoryItem]:
        """
        Converts trajectories into memory items.
        """
        prompts = []

        for t in trajectories:
            if t["success"]:
                prompt = SUCCESS_PROMPT.format(
                    question=t["question"],
                    reasoning=t["reasoning"],
                )
            else:
                prompt = FAILURE_PROMPT.format(
                    question=t["question"],
                    reasoning=t["reasoning"],
                    expected=t["expected_answer"],
                )
            prompts.append(prompt)

        outputs = self.llm.generate(prompts)

        memories: List[MemoryItem] = []

        for traj, out in zip(trajectories, outputs):
            memories.extend(
                self._parse_memory_items(
                    response=out["response"],
                    problem_id=traj["id"],
                    success=traj["success"],
                )
            )

        return memories

    def _parse_memory_items(
        self,
        response: str,
        problem_id: str,
        success: bool,
    ) -> List[MemoryItem]:
        """
        Parses raw LLM output into MemoryItem objects.
        """
        items: List[MemoryItem] = []

        blocks = re.split(r"\n\s*MEMORY\s+\d+\s*:\s*", response)

        for block in blocks[1:]:
            title = self._extract_field(block, "TITLE:")
            description = self._extract_field(block, "DESCRIPTION:")
            content = self._extract_field(block, "CONTENT:")

            if title and description and content:
                items.append(
                    MemoryItem(
                        title=title,
                        description=description,
                        content=content,
                        source_problem_id=problem_id,
                        success=success,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )

        return items

    def _extract_field(self, text: str, field: str) -> str:
        if field not in text:
            return ""

        start = text.index(field) + len(field)
        tail = text[start:]

        markers = ["TITLE:", "DESCRIPTION:", "CONTENT:"]
        end = len(text)

        for m in markers:
            idx = tail.find(m)
            if idx != -1:
                end = min(end, start + idx)

        return text[start:end].strip()


# =========================================================
# MAIN METHOD
# =========================================================

class ReasoningBank(BaseMethod):
    """
    Memory-augmented reasoning system with retrieval and self-improvement.
    """

    def __init__(
        self,
        llm,
        embed_model: EmbeddingModel,
        storage_path: str = "data/reasoning_bank.json",
        load: bool = True,
        top_k: int = 2,
        max_new_tokens: int = 1024,
        logging: Optional[str] = None,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.storage_path = storage_path
        self.memories: List[MemoryItem] = []
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.logging = logging

        if load:
            self.load_or_raise()

    def load_or_raise(self) -> None:
        path = Path(self.storage_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reasoning bank not found: {self.storage_path}"
            )
        self.load()

    def add_memories(self, memories: List[MemoryItem]) -> None:
        self.memories.extend(memories)
        self.save()

    def get_all_memories(self) -> List[MemoryItem]:
        return self.memories

    def save(self) -> None:
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in self.memories], f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.memories = [MemoryItem.from_dict(m) for m in data]
        except FileNotFoundError:
            self.memories = []
        except Exception as e:
            raise ValueError(f"Failed to load memory bank: {e}")

    def clear(self) -> None:
        self.memories = []
        self.save()

    def __len__(self) -> int:
        return len(self.memories)

    def _log(self, msg: str) -> None:
        if not self.logging:
            return
        with open(self.logging, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _extract_core_concept(self, questions: List[str]) -> List[str]:
        prompts = [KEY_CONCEPT_EXTRACTION_PROMPT.format(question=q) for q in questions]
        outputs = self.llm.generate(prompts)
        return [o["response"] for o in outputs]

    def _make_solver_prompt(self, question: str, memories: str | None = None) -> str:
        if memories:
            return QUESTION_PROMPT_WITH_MEMORIES.format(
                question=question,
                memories=memories,
            )
        return QUESTION_PROMPT.format(question=question)

    def fit(
        self,
        equations: List[str],
        solutions: List[str],
        batch_size: int = 4,
    ) -> "ReasoningBank":
        if len(equations) != len(solutions):
            raise ValueError("Inputs must have the same length")

        retriever = MemoryRetriever(self.embed_model)
        extractor = MemoryExtractor(self.llm)
        judge = LLMJudge(self.llm)

        self._log("=== FIT START ===")

        for start in tqdm(range(0, len(equations), batch_size), desc="Fit"):
            batch_eqs = equations[start:start + batch_size]
            batch_refs = solutions[start:start + batch_size]

            concepts = self._extract_core_concept(batch_eqs)

            memory_contexts = [
                retriever.format_memories_for_prompt(
                    retriever.retrieve(c, self.memories, self.top_k)
                )
                for c in concepts
            ]

            prompts = [
                self._make_solver_prompt(eq, mem)
                for eq, mem in zip(batch_eqs, memory_contexts)
            ]

            responses = self.llm.generate(prompts, max_new_tokens=self.max_new_tokens)
            predicted = [extract_boxed(r["response"]) or "" for r in responses]

            judged = judge.judge_batch(
                batch_eqs,
                predicted,
                batch_refs,
                max_new_tokens=16,
            )

            trajectories = [
                {
                    "id": f"eq_{start + i}",
                    "question": eq,
                    "reasoning": pred,
                    "expected_answer": ref,
                    "success": bool(ok),
                }
                for i, (eq, pred, ref, ok) in enumerate(
                    zip(batch_eqs, predicted, batch_refs, judged)
                )
            ]

            new_memories = extractor.extract_from_trajectories(trajectories)
            self.memories.extend(new_memories)

            self._log(f"Batch {start}: +{len(new_memories)} memories")

        self.save()
        self._log("=== FIT END ===")

        return self

    def solve(
        self,
        equations: List[str],
        batch_size: int = 4,
    ) -> List[Dict]:
        if not equations:
            return []

        retriever = MemoryRetriever(self.embed_model)

        token_counts = [0] * len(equations)
        times = [0.0] * len(equations)
        answers = [""] * len(equations)

        self._log("=== SOLVE START ===")

        for start in tqdm(range(0, len(equations), batch_size), desc="Solve"):
            batch = equations[start:start + batch_size]

            concept_outputs = self.llm.generate(
                [KEY_CONCEPT_EXTRACTION_PROMPT.format(question=q) for q in batch]
            )

            concepts = [o["response"] for o in concept_outputs]

            memory_contexts = [
                retriever.format_memories_for_prompt(
                    retriever.retrieve(c, self.memories, self.top_k)
                )
                for c in concepts
            ]

            prompts = [
                self._make_solver_prompt(eq, mem)
                for eq, mem in zip(batch, memory_contexts)
            ]

            outputs = self.llm.generate(prompts, max_new_tokens=self.max_new_tokens)

            for i, o in enumerate(outputs):
                idx = start + i
                token_counts[idx] += o["token_count"]
                times[idx] += o["avg_time"]
                answers[idx] = o["response"].strip()

            self._log(f"Batch {start} completed")

        self._log("=== SOLVE END ===")

        return [
            {
                "response": answers[i],
                "token_count": token_counts[i],
                "avg_time": times[i],
            }
            for i in range(len(equations))
        ]
