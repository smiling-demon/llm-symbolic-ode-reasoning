import json
import time
import torch
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Union, List, Tuple, Dict
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, logging


SUCCESS_PROMPT = """You successfully solved a math problem.

Your task is to extract GENERAL and ABSTRACT strategies that led to success.

The strategies must be applicable to a BROAD CLASS of math problems.

IMPORTANT RULES (MUST FOLLOW):
- Output ONLY the structured memories.
- Do NOT include any numbers, equations, variable names, or symbols.
- Do NOT include concrete examples.
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

FAILURE_PROMPT = """You attempted a math problem but the solution was incorrect.

Your task is to extract GENERAL lessons about what went wrong.

The lessons must apply to a BROAD CLASS of math problems.

IMPORTANT RULES (MUST FOLLOW):
- Output ONLY the structured memories.
- Do NOT include any numbers, equations, variable names, or symbols.
- Do NOT include concrete examples.
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

QUESTION_PROMPT = """You are a highly precise reasoning assistant. Solve the problem step by step.

Problem: {question}

INSTRUCTIONS (READ CAREFULLY):
1. Show your step-by-step reasoning in **numbered steps**, e.g.:
   Step 1: ...
   Step 2: ...
   Final answer: ...
3. THE FINAL ANSWER MUST BE STRICTLY AN INTEGER NUMBER. NO UNITS, NO WORDS, NO SYMBOLS, NO LATEX.
4. THE FINAL ANSWER MUST BE OUTPUT IN THIS EXACT FORMAT:
   Final answer: #### <final_number>
   Replace <final_number> with only the numeric answer.
5. DO NOT INCLUDE ANYTHING ELSE AFTER '####'.
6. IF YOU CANNOT CALCULATE, OUTPUT ONLY '#### NO ANSWER' AS PLACEHOLDER.

BEGIN STEP-BY-STEP REASONING:
"""

QUESTION_PROMPT_WITH_MEMORIES = """You are a highly precise reasoning assistant. Solve the problem step by step.

Problem: {question}

INSTRUCTIONS (READ CAREFULLY):
1. Show your step-by-step reasoning in **numbered steps**, e.g.:
   Step 1: ...
   Step 2: ...
   Final answer: ...
3. THE FINAL ANSWER MUST BE STRICTLY AN INTEGER NUMBER. NO UNITS, NO WORDS, NO SYMBOLS, NO LATEX.
4. THE FINAL ANSWER MUST BE OUTPUT IN THIS EXACT FORMAT:
   Final answer: #### <final_number>
   Replace <final_number> with only the numeric answer.
5. DO NOT INCLUDE ANYTHING ELSE AFTER '####'.
6. IF YOU CANNOT CALCULATE, OUTPUT ONLY '#### NO ANSWER' AS PLACEHOLDER.

If strategies below are relevant for current question, then you can use this strategies to improve solving process. You should IGNORE non-relevant strategies.
{memories}

BEGIN STEP-BY-STEP REASONING:
"""

KEY_CONCEPT_EXTRACTION_PROMPT = """You are an AI assistant tasked with extracting the essential ideas from a given problem. 
Do NOT solve the problem. Instead, identify and summarize the following:

1. **Core Concept**: The fundamental idea or principle behind the problem.
2. **Key Solution Ideas**: The abstract strategies or approaches that would be relevant for solving this type of problem, without actually performing any calculations or providing a solution.

Output format:
- Core Concept: [one or two sentences summarizing the main idea]
- Key Solution Ideas: [bullet points or short sentences, abstract and general, no calculations]

Example:
Problem: "A car accelerates from 0 to 60 km/h in 5 seconds. What is its acceleration?"
Output:
- Core Concept: Kinematics and the relationship between velocity, acceleration, and time.
- Key Solution Ideas: 
  - Use basic motion formulas relating acceleration, velocity, and time.
  - Consider units and conversion if needed.
  - Apply abstract reasoning about how change in speed relates to acceleration.

Now process the following problem:
{question}
"""


class LLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompts: Union[str, List[str]], max_new_tokens: int = 512) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        texts = [tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]

        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
    
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            # do_sample=False
        )
    
        responses = []
        for i in range(len(generated_ids)):
            output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            responses.append(response)
        return responses
        

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
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class ReasoningBank:
    def __init__(self, storage_path='memory_bank/reasoning_bank.json'):
        self.storage_path = storage_path
        self.memories: List[MemoryItem] = []
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
    
        with open(self.storage_path, 'w') as f:
            data = [m.to_dict() for m in self.memories]
            json.dump(data, f, indent=2)
    
    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
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
    def __init__(self, embedding_model):
        self.model = embedding_model
    
    def embed_memories(self, memories: List["MemoryItem"]):
        texts = [f"{m.title}. {m.description}" for m in memories]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        for memory, embedding in zip(memories, embeddings):
            memory.embedding = embedding.tolist()
    
    def retrieve(
        self, 
        query: str,
        memories: List["MemoryItem"],
        top_k: int = 2,
        random_k: int = 1,
        similarity_threshold: float = 0.95
    ) -> List[Tuple["MemoryItem", float]]:
        if not memories:
            return []
    
        if any(m.embedding is None for m in memories):
            self.embed_memories(memories)
    
        query_embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        memory_embeddings = np.array([m.embedding for m in memories])

        similarities = np.dot(memory_embeddings, query_embedding)
    
        candidate_indices = np.argsort(similarities)[-2 * top_k:][::-1]
    
        selected = []
        selected_embeddings = []
    
        for idx in candidate_indices:
            emb = memory_embeddings[idx]
            sim = float(similarities[idx])
    
            if not selected_embeddings:
                selected.append((memories[idx], sim))
                selected_embeddings.append(emb)
                continue
    
            max_sim = max(np.dot(emb, sel_emb) for sel_emb in selected_embeddings)
    
            if max_sim < similarity_threshold:
                selected.append((memories[idx], sim))
                selected_embeddings.append(emb)
    
            if len(selected) >= top_k:
                break
    
        seen_titles = {m.title for m, s in selected}
        remaining = [m for m in memories if m.title not in seen_titles]
    
        if remaining and random_k > 0:
            random_memories = random.sample(remaining, min(random_k, len(remaining)))
            for m in random_memories:
                selected.append((m, 0.0))
    
        return selected
    
    def format_memories_for_prompt(self, retrieved: List[Tuple["MemoryItem", float]]) -> str:
        if not retrieved:
            return ""
        formatted = "Past Strategy Hints:\n\n"
        formatted += "IMPORTANT: These are STRATEGY hints only. Do NOT copy any numbers from them.\n\n"
        for idx, (m, score) in enumerate(retrieved, 1):
            status = "Success Strategy" if m.success else "Lesson from Failure"
            formatted += f"## Strategy {idx} ({status}):\n**{m.title}**\n{m.content}\n\n"
        return formatted
        
        
class MemoryExtractor:
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract_from_trajectories(self, trajectories: List[Dict]) -> List[List[MemoryItem]]:
        prompts = []
        for trajectory in trajectories:
            question = trajectory['question']
            reasoning = trajectory['reasoning']
            expected = trajectory['expected_answer']
            if trajectory['success']:
                prompt = SUCCESS_PROMPT.format(question=question, reasoning=reasoning)
            else:
                prompt = FAILURE_PROMPT.format(question=question, reasoning=reasoning, expected=expected)
            prompts.append(prompt)

        responses = self.llm.generate(prompts)

        results: List[List[MemoryItem]] = []
        for trajectory, response in zip(trajectories, responses):
            mems = self._parse_memory_items(response, trajectory['id'], trajectory['success'])
            results.extend(mems)
        return results

    def _parse_memory_items(self, response: str, problem_id: str, success: bool) -> List[MemoryItem]:
        memories: List[MemoryItem] = []
        parts = response.split('MEMORY')
        for part in parts[1:]:
            try:
                title = self._extract_field(part, 'TITLE:')
                description = self._extract_field(part, 'DESCRIPTION:')
                content = self._extract_field(part, 'CONTENT:')
                if title and description and content:
                    memory = MemoryItem(
                        title=title,
                        description=description,
                        content=content,
                        source_problem_id=problem_id,
                        success=success,
                        created_at=datetime.now().isoformat()
                    )
                    memories.append(memory)
            except Exception:
                continue
        return memories

    def _extract_field(self, text: str, field_name: str) -> str:
        if field_name not in text:
            return ""
        start = text.index(field_name) + len(field_name)
        next_markers = ['TITLE:', 'DESCRIPTION:', 'CONTENT:', 'MEMORY', '\n\nMEMORY', '\nMEMORY']
        end = len(text)
        tail = text[start:]
        for m in next_markers:
            idx = tail.find(m)
            if idx != -1:
                candidate_end = start + idx
                if candidate_end < end:
                    end = candidate_end
        value = text[start:end].strip()
        return value
        

class Experiment:
    def __init__(self, llm_model, llm_tokenizer, embed_model, storage_path):
        self.llm = LLM(llm_model, llm_tokenizer)
        self.memory_bank = ReasoningBank(storage_path)
        self.retriever = MemoryRetriever(embed_model)
        self.extractor = MemoryExtractor(self.llm)

    def _make_solver_prompt(self, question: str, memories_text: Optional[str] = None) -> str:
        if memories_text:
            return QUESTION_PROMPT_WITH_MEMORIES.format(question=question, memories=memories_text)
        return QUESTION_PROMPT.format(question=question)

    def _make_strategy_prompt(self, question: str) -> str:
        return KEY_CONCEPT_EXTRACTION_PROMPT.format(question=question)

    def _extract_reasoning_and_answer_from_text(self, text: str):
        if '####' in text:
            parts = text.split('####')
            reasoning = "###".join(parts[:-1])
            answer = parts[-1].strip()
            return reasoning, answer
        return text, ""
    
    def solve_batch(
        self,
        questions: List[str],
        memories_texts: Optional[List[Optional[str]]] = None,
    ) -> List[Tuple[str, str]]:
        if memories_texts is None:
            memories_texts = [None] * len(questions)
    
        prompts = [
            self._make_solver_prompt(q, m)
            for q, m in zip(questions, memories_texts)
        ]
    
        responses = self.llm.generate(prompts)
    
        outputs = []
        for response in responses:
            reasoning, answer = self._extract_reasoning_and_answer_from_text(response)
            outputs.append((reasoning, answer))
    
        return outputs
    
    def extract_strategies(
        self,
        questions: List[str],
    ) -> List[Tuple[str, str]]:
        prompts = [
            self._make_strategy_prompt(q)
            for q in questions
        ]
    
        concepts = self.llm.generate(prompts)
    
        return concepts

    def retrieve_memory_contexts(self, batch, concepts, top_k, random_k):
        memory_contexts = []
    
        for problem, concept in zip(batch, concepts):
            retrieved = self.retriever.retrieve(
                concept,
                self.memory_bank.get_all_memories(),
                top_k=top_k,
                random_k=random_k
            )
            memory_context = self.retriever.format_memories_for_prompt(retrieved)

            memory_contexts.append(memory_context)

        return memory_contexts

    def train_memory_bank(
        self,
        problems: List[Dict],
        batch_size: int = 10,
        top_k: int = 2,
        random_k: int = 1
    ):
        print("Training memory bank...")
    
        for start in tqdm(range(0, len(problems), batch_size), desc="Training"):
            batch = problems[start:start + batch_size]
    
            questions = [problem["question"] for problem in batch]

            concepts = self.extract_strategies(questions)
            memory_contexts = self.retrieve_memory_contexts(batch, concepts, top_k, random_k)
    
            solved = self.solve_batch(questions, memory_contexts)
    
            trajectories = []
            for problem, (reasoning, predicted_answer) in zip(batch, solved):
                success = predicted_answer == problem["answer"]
    
                trajectories.append({
                    "id": problem["id"],
                    "question": problem["question"],
                    "expected_answer": problem["answer"],
                    "reasoning": reasoning,
                    "predicted_answer": predicted_answer,
                    "success": success,
                })

            memories = self.extractor.extract_from_trajectories(trajectories)
            self.memory_bank.add_memories(memories)
    
        print(f"Memory bank contains: {len(self.memory_bank)} items\n")


    def evaluate_memory_bank(self, problems: List[Dict], batch_size: int = 50, top_k: int = 2, random_k: int = 1, use_memory=True):
        accuracy = 0
    
        for start in tqdm(range(0, len(problems), batch_size), desc="Training"):
            batch = problems[start:start + batch_size]
    
            questions = [problem["question"] for problem in batch]
            expected_answers = [problem["answer"] for problem in batch]

            memory_contexts = None
            
            if use_memory:
                concepts = self.extract_strategies(questions)
                memory_contexts = self.retrieve_memory_contexts(batch, concepts, top_k, random_k)
    
            solved = self.solve_batch(questions, memory_contexts)

            for problem, (reasoning, predicted_answer), expected_answer in zip(batch, solved, expected_answers):
                success = predicted_answer == expected_answer
                accuracy += success / len(problems)

        return accuracy

