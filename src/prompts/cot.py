COT_PROMPT_TEMPLATE = """Solve the following differential equation concisely with essential derivation steps only.

Problem (in LaTeX):
{question}

INSTRUCTIONS:
1. Solve step by step for y(x).
2. Keep the solution concise and avoid obvious transitions or verbose explanations.
3. Use valid LaTeX notation.
4. The final solution MUST be in the form:

Final answer: \\boxed{{y(x) = <solution>}}

5. Do NOT write anything after the boxed expression.

BEGIN SOLUTION:
"""