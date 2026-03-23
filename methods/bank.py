from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from metrics import exact_match
from utils import extract_boxed


SUCCESS_PROMPT = """You successfully solved a differential equation.

Your task is to extract GENERAL and ABSTRACT strategies that led to success.

The strategies must be applicable to a BROAD CLASS of differential equation problems.

IMPORTANT RULES (MUST FOLLOW):
- Output ONLY the structured memories.
- Do NOT include concrete numbers, coefficients, or specific equations.
- Do NOT include variable names or symbols from the problem.
- Do NOT include worked examples.
- Strategies must be fully abstract and reusable.
- Use EXACTLY the format shown below.

PROBLEM (FOR CONTEXT ONLY — DO NOT MENTION):
{question}

MODEL SOLUTION (FOR CONTEXT ONLY — DO NOT MENTION):
{reasoning}

OUTPUT FORMAT (COPY EXACTLY):

MEMORY 1:
TITLE: <abstract strategy name>
DESCRIPTION: <one general sentence>
CONTENT: <detailed, abstract, reusable reasoning strategy>

MEMORY 2:
TITLE: <abstract strategy name>
DESCRIPTION: <one general sentence>
CONTENT: <detailed, abstract, reusable reasoning strategy>

(OPTIONAL) MEMORY 3:
TITLE: <abstract strategy name>
DESCRIPTION: <one general sentence>
CONTENT: <detailed, abstract, reusable reasoning strategy>

END OF OUTPUT.
"""


FAILURE_PROMPT = """You attempted to solve a differential equation, but the solution was incorrect.

Your task is to extract GENERAL lessons about what went wrong.

The lessons must apply to a BROAD CLASS of differential equation problems.

IMPORTANT RULES (MUST FOLLOW):
- Output ONLY the structured memories.
- Do NOT include concrete numbers, coefficients, or specific equations.
- Do NOT include variable names or symbols from the problem.
- Do NOT include worked examples.
- Lessons must be abstract and preventive.
- Use EXACTLY the format shown below.

PROBLEM (FOR CONTEXT ONLY — DO NOT MENTION):
{question}

MODEL ATTEMPT (FOR CONTEXT ONLY — DO NOT MENTION):
{reasoning}

EXPECTED ANSWER (FOR CONTEXT ONLY — DO NOT MENTION):
{expected}

OUTPUT FORMAT (COPY EXACTLY):

MEMORY 1:
TITLE: <abstract mistake or check>
DESCRIPTION: <one general sentence>
CONTENT: <abstract lesson or preventive strategy>

MEMORY 2:
TITLE: <abstract mistake or check>
DESCRIPTION: <one general sentence>
CONTENT: <abstract lesson or preventive strategy>

(OPTIONAL) MEMORY 3:
TITLE: <abstract mistake or check>
DESCRIPTION: <one general sentence>
CONTENT: <abstract lesson or preventive strategy>

END OF OUTPUT.
"""


QUESTION_PROMPT = """You are a precise mathematical assistant specialized in solving differential equations.

Solve the following differential equation step by step.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
1. Provide a clear step-by-step solution.
2. Be concise and don't write obvious transformations.
3. The final solution MUST be written in LaTeX.
4. The FINAL answer MUST be in the format:

Final answer: \\boxed{{<solution>}}

5. Do NOT include anything after the boxed expression.

BEGIN SOLUTION:
"""


QUESTION_PROMPT_WITH_MEMORIES = """You are a precise mathematical assistant specialized in solving differential equations.

Solve the following differential equation step by step.

Problem (in LaTeX):
{question}

If strategies below are relevant for the current problem, use them to improve the solving process.
Ignore non-relevant strategies.

{memories}

INSTRUCTIONS:
1. Provide a clear step-by-step solution.
2. Show intermediate transformations and reasoning.
3. The final solution MUST be written in LaTeX.
4. The FINAL answer MUST be in the format:

Final answer: \\boxed{{<solution>}}

5. Do NOT include anything after the boxed expression.

BEGIN SOLUTION:
"""


KEY_CONCEPT_EXTRACTION_PROMPT = """You are an AI assistant extracting the essential ideas from a differential equation problem.

Do NOT solve the problem.
Instead, identify the main concept and the abstract solution ideas that would be relevant.

Output format:
- Core Concept: [one or two sentences]
- Key Solution Ideas: [bullet points or short sentences, abstract and general, no calculations]

Problem:
{question}
"""


@dataclass
class MemoryItem:
    title: str
    description: str
    content: str
    source_problem_id: str
    success: bool
    created_at: str
    embedding: Optional[List[float]] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class ReasoningBank:
    def __init__(self, storage_path: str = "memory_bank/reasoning_bank.json", load: bool = True):
        self.storage_path = storage_path
        self.memories: List[MemoryItem] = []

        if load:
            self.load_or_raise()

    def load_or_raise(self):
        path = Path(self.storage_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reasoning bank not found at '{self.storage_path}'. "
                f"Train it first with train_reasoning_bank(...)."
            )
        self.load()

    def add_memory(self, memory: MemoryItem):
        self.memories.append(memory)
        self.save()

    def add_memories(self, memories: List[MemoryItem]):
        self.memories.extend(memories)
        self.save()

    def get_all_memories(self) -> List[MemoryItem]:
        return self.memories

    def save(self):
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in self.memories], f, indent=2, ensure_ascii=False)

    def load(self):
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.memories = [MemoryItem.from_dict(m) for m in data]
        except FileNotFoundError:
            self.memories = []

    def clear(self):
        self.memories = []
        self.save()

    def __len__(self):
        return len(self.memories)


class MemoryRetriever:
    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model

    def embed_memories(self, memories: List[MemoryItem]):
        texts = [f"{m.title}. {m.description}. {m.content}" for m in memories]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        for memory, embedding in zip(memories, embeddings):
            memory.embedding = embedding.tolist()

    def retrieve(
        self,
        query: str,
        memories: List[MemoryItem],
        top_k: int = 2,
        similarity_threshold: float = 0.95,
    ) -> List[Tuple[MemoryItem, float]]:
        if not memories:
            return []

        if any(m.embedding is None for m in memories):
            self.embed_memories(memories)

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        memory_embeddings = np.array([m.embedding for m in memories], dtype=np.float32)
        similarities = memory_embeddings @ query_embedding

        candidate_indices = np.argsort(similarities)[-2 * top_k:][::-1]

        selected: List[Tuple[MemoryItem, float]] = []
        selected_embeddings = []

        for idx in candidate_indices:
            emb = memory_embeddings[idx]
            sim = float(similarities[idx])

            if not selected_embeddings:
                selected.append((memories[idx], sim))
                selected_embeddings.append(emb)
                continue

            max_sim = max(float(np.dot(emb, sel_emb)) for sel_emb in selected_embeddings)

            if max_sim < similarity_threshold:
                selected.append((memories[idx], sim))
                selected_embeddings.append(emb)

            if len(selected) >= top_k:
                break

        return selected

    def format_memories_for_prompt(self, retrieved: List[Tuple[MemoryItem, float]]) -> str:
        if not retrieved:
            return ""

        formatted = "Past Strategy Hints:\n\n"
        formatted += "IMPORTANT: These are strategy hints only. Do NOT copy any numbers from them.\n\n"
        for idx, (m, score) in enumerate(retrieved, 1):
            status = "Success Strategy" if m.success else "Lesson from Failure"
            formatted += f"## Strategy {idx} ({status}):\n"
            formatted += f"Title: {m.title}\n"
            formatted += f"{m.content}\n\n"
        return formatted


class MemoryExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract_from_trajectories(self, trajectories: List[Dict]) -> List[MemoryItem]:
        prompts = []
        for trajectory in trajectories:
            if trajectory["success"]:
                prompt = SUCCESS_PROMPT.format(
                    question=trajectory["question"],
                    reasoning=trajectory["reasoning"],
                )
            else:
                prompt = FAILURE_PROMPT.format(
                    question=trajectory["question"],
                    reasoning=trajectory["reasoning"],
                    expected=trajectory["expected_answer"],
                )
            prompts.append(prompt)

        responses = self.llm.generate(prompts)

        results: List[MemoryItem] = []
        for trajectory, response in zip(trajectories, responses):
            mems = self._parse_memory_items(response, trajectory["id"], trajectory["success"])
            results.extend(mems)
        return results

    def _parse_memory_items(self, response: str, problem_id: str, success: bool) -> List[MemoryItem]:
        memories: List[MemoryItem] = []
        parts = response.split("MEMORY")
        for part in parts[1:]:
            try:
                title = self._extract_field(part, "TITLE:")
                description = self._extract_field(part, "DESCRIPTION:")
                content = self._extract_field(part, "CONTENT:")
                if title and description and content:
                    memories.append(
                        MemoryItem(
                            title=title,
                            description=description,
                            content=content,
                            source_problem_id=problem_id,
                            success=success,
                            created_at=datetime.now().isoformat(),
                        )
                    )
            except Exception:
                continue
        return memories

    def _extract_field(self, text: str, field_name: str) -> str:
        if field_name not in text:
            return ""
        start = text.index(field_name) + len(field_name)
        next_markers = ["TITLE:", "DESCRIPTION:", "CONTENT:", "MEMORY", "\n\nMEMORY", "\nMEMORY"]
        end = len(text)
        tail = text[start:]
        for marker in next_markers:
            idx = tail.find(marker)
            if idx != -1:
                end = min(end, start + idx)
        return text[start:end].strip()


def _make_solver_prompt(question: str, memories_text: Optional[str] = None) -> str:
    if memories_text:
        return QUESTION_PROMPT_WITH_MEMORIES.format(question=question, memories=memories_text)
    return QUESTION_PROMPT.format(question=question)


def _extract_core_concept(llm, equations: List[str]) -> List[str]:
    prompts = [KEY_CONCEPT_EXTRACTION_PROMPT.format(question=q) for q in equations]
    return llm.generate(prompts)


def train_reasoning_bank(
    llm,
    embed_model: SentenceTransformer,
    equations: List[str],
    solutions: List[str],
    storage_path: str = "data/reasoning_bank.json",
    batch_size: int = 8,
    top_k: int = 2,
    max_new_tokens: int = 1024,
) -> ReasoningBank:
    assert len(equations) == len(solutions), "equations and solutions must have the same length"

    bank = ReasoningBank(storage_path=storage_path, load=False)
    retriever = MemoryRetriever(embed_model)
    extractor = MemoryExtractor(llm)

    for start in range(0, len(equations), batch_size):
        batch_eqs = equations[start:start + batch_size]
        batch_refs = solutions[start:start + batch_size]
        batch_ids = [f"eq_{start + i}" for i in range(len(batch_eqs))]

        concepts = _extract_core_concept(llm, batch_eqs)

        memory_contexts = []
        for eq, concept in zip(batch_eqs, concepts):
            retrieved = retriever.retrieve(
                query=concept,
                memories=bank.get_all_memories(),
                top_k=top_k
            )
            memory_contexts.append(retriever.format_memories_for_prompt(retrieved))

        prompts = [
            _make_solver_prompt(eq, mem)
            for eq, mem in zip(batch_eqs, memory_contexts)
        ]

        responses = llm.generate(prompts, max_new_tokens=max_new_tokens)

        trajectories = []
        for problem_id, eq, pred, ref in zip(batch_ids, batch_eqs, responses, batch_refs):
            trajectories.append(
                {
                    "id": problem_id,
                    "question": eq,
                    "reasoning": pred,
                    "expected_answer": ref,
                    "success": exact_match(extract_boxed(pred), ref) if extract_boxed(pred) else False,
                }
            )

        new_memories = extractor.extract_from_trajectories(trajectories)
        bank.add_memories(new_memories)

    bank.save()
    return bank


def load_reasoning_bank(storage_path: str = "memory_bank/reasoning_bank.json") -> ReasoningBank:
    return ReasoningBank(storage_path=storage_path, load=True)


def solve_with_reasoning_bank(
    bank: ReasoningBank,
    llm,
    embed_model: SentenceTransformer,
    equations: List[str],
    batch_size: int = 8,
    top_k: int = 2,
    max_new_tokens: int = 1024,
) -> List[str]:
    if bank is None:
        raise ValueError(
            "ReasoningBank is required. Train or load a bank first, then pass it here."
        )

    retriever = MemoryRetriever(embed_model)
    concepts = _extract_core_concept(llm, equations)

    memory_contexts = []
    for eq, concept in zip(equations, concepts):
        retrieved = retriever.retrieve(
            query=concept,
            memories=bank.get_all_memories(),
            top_k=top_k
        )
        memory_contexts.append(retriever.format_memories_for_prompt(retrieved))

    prompts = [
        _make_solver_prompt(eq, mem)
        for eq, mem in zip(equations, memory_contexts)
    ]

    output = []

    for start in range(0, len(prompts), batch_size):
        output += llm.generate(prompts[start:start + batch_size], max_new_tokens=max_new_tokens)

    return output
