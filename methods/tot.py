from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from methods.base import BaseMethod


DEFAULT_MAX_DEPTH = 6
DEFAULT_BRANCHING_FACTOR = 2
DEFAULT_MAX_STATES = 4


FINAL_ANSWER_RE = re.compile(
    r"^\s*Final answer:\s*\\boxed\{(?P<answer>.*)\}\s*$",
    re.DOTALL,
)

NUMBERED_OR_BULLET_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*(.+?)\s*$")



EXPANSION_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Problem (in LaTeX):
{question}

Current solution transcript:
{transcript}

Task:
Produce exactly ONE meaningful next step for continuing the solution.

Requirements:
1. The step should be substantial and mathematically meaningful.
2. It may include multiple algebraic operations if appropriate.
3. Do not make the step trivial or overly short.
4. Do not write more than one step.
5. If the this step is final and solution can be completed, the step MUST be written exactly in this form:
   Final answer: \\boxed{{<solution>}}
6. Do not add any extra text before or after the step.

Return ONLY the next step.
"""

PRUNE_PROMPT_TEMPLATE = """You are ranking intermediate solution states for a differential equation.

Problem (in LaTeX):
{question}

Choose the BEST states that are closest to a correct final solution.

Prefer states that:
1. Are mathematically valid,
2. Make real progress,
3. Stay consistent with the problem,
4. Look more promising for reaching a correct final answer.

Return ONLY a JSON list of indices, ordered from best to worst.
Example:
[2, 0, 4, 1, 3]

Candidates:
{candidates}
"""

FINAL_SELECTION_PROMPT_TEMPLATE = """You are selecting the best completed solution for a differential equation.

Problem (in LaTeX):
{question}

We have several completed candidates. Each one should contain a final answer in the exact form:
Final answer: \\boxed{{<solution>}}

Select the SINGLE best candidate by correctness and clarity.

Return ONLY the index of the best candidate as an integer.
Example:
2

Candidates:
{candidates}
"""

FINALIZE_PROMPT_TEMPLATE = """You are a precise mathematical assistant specialized in solving differential equations.

Problem (in LaTeX):
{question}

Current solution transcript:
{transcript}

Task:
Continue from the transcript and output ONLY the final answer in the exact format:

Final answer: \\boxed{<solution>}

Rules:
1. Do not write any explanation.
2. Do not add anything after the boxed expression.
3. Ensure the answer is in LaTeX.

BEGIN FINAL ANSWER:
"""


def extract_final_answer(text: str) -> Optional[str]:
    match = FINAL_ANSWER_RE.match(text.strip())
    if not match:
        return None
    return match.group("answer").strip()


def is_final_step(text: str) -> bool:
    return extract_final_answer(text) is not None


def clean_step_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    m = NUMBERED_OR_BULLET_RE.match(line)
    if m:
        return m.group(1).strip()
    return line


def parse_single_step(text: str) -> str:
    lines = [clean_step_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return text.strip()
    return lines[0].strip()


def parse_ranking_indices(text: str, n: int) -> List[int]:
    raw = text.strip()

    try:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            data = json.loads(candidate)
            if isinstance(data, list):
                out: List[int] = []
                for x in data:
                    try:
                        idx = int(x)
                        if 0 <= idx < n and idx not in out:
                            out.append(idx)
                    except Exception:
                        pass
                if out:
                    return out
    except Exception:
        pass

    out = []
    for m in re.finditer(r"\d+", raw):
        idx = int(m.group())
        if 0 <= idx < n and idx not in out:
            out.append(idx)
    return out


def parse_best_index(text: str, n: int) -> Optional[int]:
    raw = text.strip()

    m = re.search(r"\d+", raw)
    if m:
        idx = int(m.group())
        if 0 <= idx < n:
            return idx

    return None


def dedupe_states(states: List["State"]) -> List["State"]:
    seen = set()
    out: List["State"] = []
    for s in states:
        key = s.render().strip()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


@dataclass
class State:
    question: str
    steps: List[str] = field(default_factory=list)
    finished: bool = False
    final_answer: Optional[str] = None

    def render(self) -> str:
        if not self.steps:
            return "No steps yet."
        return "\n".join(self.steps)

    def copy(self) -> "State":
        return State(
            question=self.question,
            steps=list(self.steps),
            finished=self.finished,
            final_answer=self.final_answer,
        )

    def add_step(self, step: str) -> None:
        step = step.strip()
        if not step:
            return
        self.steps.append(step)
        ans = extract_final_answer(step)
        if ans is not None:
            self.finished = True
            self.final_answer = ans


class ToT(BaseMethod):
    def __init__(
        self,
        llm,
        max_new_tokens: int = 512,
        max_depth: int = DEFAULT_MAX_DEPTH,
        branching_factor: int = DEFAULT_BRANCHING_FACTOR,
        max_states: int = DEFAULT_MAX_STATES,
    ):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.max_states = max_states


    def _build_expansion_prompts(self, states: List[State]) -> List[str]:
        prompts: List[str] = []
        for state in states:
            for branch_id in range(self.branching_factor):
                prompts.append(
                    EXPANSION_PROMPT_TEMPLATE.format(
                        question=state.question,
                        transcript=state.render(),
                    )
                    + f"\n\nVariant hint: {branch_id + 1}"
                )
        return prompts

    def _build_prune_prompt(self, states: List[State]) -> str:
        return PRUNE_PROMPT_TEMPLATE.format(
            question=states[0].question if states else "",
            candidates="\n\n".join(
                f"[{i}]\n{s.render()}" for i, s in enumerate(states)
            ),
        )

    def _build_final_selection_prompt(self, states: List[State]) -> str:
        return FINAL_SELECTION_PROMPT_TEMPLATE.format(
            question=states[0].question if states else "",
            candidates="\n\n".join(
                f"[{i}]\n{s.render()}" for i, s in enumerate(states)
            ),
        )

    def _build_finalize_prompt(self, state: State) -> str:
        return FINALIZE_PROMPT_TEMPLATE.format(
            question=state.question,
            transcript=state.render(),
        )


    def expand(self, states: List[State]) -> List[State]:

        active_states = [s for s in states if not s.finished]
        finished_states = [s for s in states if s.finished]

        if not active_states:
            return finished_states

        prompts = self._build_expansion_prompts(active_states)
        responses = self.llm.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )

        new_states: List[State] = list(finished_states)

        idx = 0
        for parent in active_states:
            for _ in range(self.branching_factor):
                if idx >= len(responses):
                    break
                raw_step = responses[idx]
                idx += 1

                step = parse_single_step(raw_step)
                child = parent.copy()
                child.add_step(step)
                new_states.append(child)

        return dedupe_states(new_states)


    def prune(self, states: List[State]) -> List[State]:
        if len(states) <= self.max_states:
            return states

        finals = [s for s in states if s.finished]
        non_finals = [s for s in states if not s.finished]

        if len(finals) > self.max_states:
            prompt = self._build_prune_prompt(finals)

            response = self.llm.generate(
                [prompt],
                max_new_tokens=256,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )[0]

            ranked = parse_ranking_indices(response, len(finals))

            if not ranked:
                finals = finals[:self.max_states]
            else:
                finals = [finals[i] for i in ranked[:self.max_states]]

            return finals

        remaining = self.max_states - len(finals)

        if remaining <= 0:
            return finals[:self.max_states]

        if len(non_finals) <= remaining:
            return finals + non_finals

        prompt = self._build_prune_prompt(non_finals)

        response = self.llm.generate(
            [prompt],
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )[0]

        ranked = parse_ranking_indices(response, len(non_finals))

        if not ranked:
            non_finals = non_finals[:remaining]
        else:
            non_finals = [non_finals[i] for i in ranked[:remaining]]

        return finals + non_finals


    def select_best_final(self, states: List[State]) -> State:
        if not states:
            raise ValueError("select_best_final() received an empty state list.")

        if len(states) == 1:
            return states[0]

        prompt = self._build_final_selection_prompt(states)

        response = self.llm.generate(
            [prompt],
            max_new_tokens=64,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )[0]

        best_idx = parse_best_index(response, len(states))
        if best_idx is None:
            return states[0]

        return states[best_idx]

    def finalize_if_needed(self, state: State) -> State:
        if state.finished:
            return state

        prompt = self._build_finalize_prompt(state)
        response = self.llm.generate(
            [prompt],
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )[0].strip()

        finalized = state.copy()
        finalized.add_step(response)

        if not finalized.finished:
            finalized.steps.append(response)

        return finalized


    def solve_one(self, equation: str) -> str:
        states = [State(question=equation)]

        for _ in range(self.max_depth):
            states = self.expand(states)
            print("EXPAND:\n", states)
            states = self.prune(states)
            print("PRUNE:\n", states)

        finals = [s for s in states if s.finished]

        if finals:
            best = self.select_best_final(finals)
            return best.render()

        if states:
            best_state = states[0]
            finalized = self.finalize_if_needed(best_state)
            return finalized.render()

        return ""

    def solve(self, equations: List[str]) -> List[str]:
        return [self.solve_one(eq) for eq in equations]


def tot(
    llm,
    equations: List[str],
    max_new_tokens: int = 512,
    max_depth: int = DEFAULT_MAX_DEPTH,
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
    max_states: int = DEFAULT_MAX_STATES,
) -> List[str]:
    return ToT(
        llm,
        max_new_tokens=max_new_tokens,
        max_depth=max_depth,
        branching_factor=branching_factor,
        max_states=max_states,
    ).solve(equations)
