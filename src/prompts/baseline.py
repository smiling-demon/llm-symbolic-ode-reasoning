BASELINE_PROMPT_TEMPLATE = """Solve the following differential equation.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
1. Do NOT write steps of solution, you need to write ONLY final answer.
1. The final solution MUST be in LaTeX.
2. Present the FINAL answer ONLY in the format:

Final answer: \\boxed{{y(x) = <solution>}}

3. Do NOT write anything after the boxed expression.

Final answer: """
