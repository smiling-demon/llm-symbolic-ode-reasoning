AGGREGATION_PROMPT_TEMPLATE = """You are given a differential equation and several candidate solutions.

Problem (in LaTeX):
{question}

Candidate solutions:
{candidates}

INSTRUCTIONS:
1. Analyze all candidate solutions carefully.
2. Keep only correct and useful reasoning.
3. If all candidates are incorrect, solve from scratch.
4. Produce one coherent step-by-step solution.
5. Keep the solution concise and avoid obvious transitions or verbose explanations.
6. The final solution MUST be written in LaTeX.
7. The FINAL answer MUST be in the format:

Final answer: \\boxed{{y(x) = <solution>}}

8. Do NOT include anything after the boxed expression.

BEGIN AGGREGATED SOLUTION:
"""
