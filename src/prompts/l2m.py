DECOMPOSITION_PROMPT_TEMPLATE = """Your task is to decompose the problem into a minimal ordered sequence of up to 3 necessary solving stages.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
- Output ONLY a numbered list of subproblems.
- Use at most 3 steps.
- Each step must be strictly necessary to solve the problem.
- Do NOT include solution attempts.
- Do NOT include formulas or symbolic derivations.
- Each step must describe a clear mathematical action.

GOOD STYLE EXAMPLES:
- identify type of differential equation
- derive characteristic equation from linear homogeneous ODE
- compute roots of auxiliary equation and classify cases

BAD STYLE:
- solve the equation
- analyze the problem
- find solution

OUTPUT FORMAT:
1. ...
2. ...
3. ...

BEGIN DECOMPOSITION:
"""


SUBPROBLEM_SOLVING_PROMPT_TEMPLATE = """Your task is to solve given subproblem.

Original problem (LaTeX):
{question}

Current subproblem:
{subproblem}

Previously completed subproblems:
{history}

INSTRUCTIONS:
- Solve ONLY the current subproblem.
- Do NOT jump ahead to later stages.
- Do NOT solve the full original problem.
- Use previous subproblems only as context.
- Keep the solution focused and concise.
- If computation is needed, show only necessary transformations.

IMPORTANT:
This is part of a multi-step pipeline. Do not attempt full final solution.

BEGIN SOLUTION:
"""


FINAL_SYNTHESIS_PROMPT_TEMPLATE = """Your task is to synthesize solution of the original problem based on solved subproblems.

Original problem (LaTeX):
{question}

Solved subproblems (ordered pipeline):
{history}

INSTRUCTIONS:
- Reconstruct the full solution using ONLY the information from solved subproblems.
- Do NOT introduce new methods or new derivations.
- Present the solution in a compact, logically continuous form.
- Remove unnecessary intermediate explanations and keep only essential mathematical steps.

OUTPUT STRUCTURE:
1. Compact solution (clean derivation using subproblem results)
2. Final answer in boxed LaTeX format.

STRICT OUTPUT FORMAT:
<compact step-by-step solution>
Final answer: \\boxed{{y(x) = <solution>}}

RULES:
- No missing steps in logic (but keep them concise).
- No repetition of full subproblem texts.
- Do not solve anything beyond given subproblem results.

BEGIN FINAL SYNTHESIS:
"""
