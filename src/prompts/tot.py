TOT_GENERATE_PROMPT = r"""Your task is to generate exactly ONE next reasoning step that represents a MEANINGFUL mathematical advance.

Problem (LaTeX):
{question}

Previous reasoning steps:
{state}

INSTRUCTIONS:
The step must be a SUBSTANTIVE transformation of the solution state, such as:
- constructing and solving the characteristic equation,
- performing a substitution (not just defining it),
- completing separation of variables,
- applying an integrating factor and simplifying the result,
- finding general solution structure.

Final-answer policy:
- Output a final answer ONLY if the previous state already contains a full solution.
- Otherwise do NOT output a final answer.

If and only if the solution is complete, then we append answer in format:
Final answer: \\boxed{{y(x)=<solution>}}

BEGIN NEXT REASONING STEP:
"""


TOT_RANK_PROMPT = """You are ranking solution trajectories for a differential equation.

Problem:
{question}

Candidates:
{candidates}

Task:
Rank ALL candidates from BEST to WORST.

Criteria (in order of importance):
1. Mathematical correctness
2. Progress toward a correct solving method
3. Use of valid differential equation techniques
4. Consistency of reasoning

Rules:
- Each candidate index MUST appear exactly once.
- Do NOT omit or duplicate indices.
- Output ONLY the ranking.
- No explanations, no extra text.

EXAMPLE:

Example candidates:
[1]
Correct and complete solution. Uses valid ODE method and reaches final correct answer.

[2]
Mostly correct approach, but contains a small algebraic mistake that makes the final answer incorrect.

[3]
Incorrect method. Applies an invalid technique and does not lead toward a valid solution.

[4]
Correct method identified, but reasoning is incomplete and stops early before solving.

Example output:
2 > 1 > 4 > 3

Now rank the given candidates:

BEGIN RANKING:
"""


TOT_FINAL_PROMPT = """Your task is to finish problem solution.

Problem (LaTeX):
{question}

Previous reasoning steps:
{state}

INSTRUCTIONS:
- Continue solving the equation ONLY if it is not already solved.
- Do NOT rewrite or repeat existing steps unnecessarily.
- Extend the current reasoning from the provided state.
- If the solution is already complete, simply output the final answer in the required format.
- Maintain mathematical correctness.
- The final answer MUST be in LaTeX.
- Do NOT write anything after final answer.

Required final answer format:
Final answer: \\boxed{{y(x) = <solution>}}

BEGIN FINISHING OF THE PROBLEM:
"""
