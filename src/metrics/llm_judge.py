import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are a strict mathematical judge.

You are given two expressions in LaTeX form.
Your task is to decide whether they represent equivalent solutions of the SAME differential equation.

Important:
- They may look different algebraically.
- Constants may be renamed or reparameterized.
- Equivalent ODE solution families must be considered equal.
- You must ignore superficial transformations.

Answer ONLY:
YES or NO

Do not explain.

Your verdict:
"""


def llm_judge(ground_truth: str, predicted_answer: str) -> float:
    """
    Evaluates whether two LaTeX expressions represent equivalent solutions
    of the same differential equation using an LLM judge.

    Args:
        ground_truth (str): Reference solution in LaTeX format.
        predicted_answer (str): Predicted solution in LaTeX format.

    Returns:
        float: 1.0 if equivalent (YES), otherwise 0.0.
    """
    if not ground_truth or not predicted_answer:
        return 0.0

    user_prompt = f"""
GROUND TRUTH:
{ground_truth}

PREDICTED:
{predicted_answer}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip().upper()

    return 1.0 if "YES" in answer else 0.0
