SUCCESS_PROMPT = """You successfully solved a differential equation.

Your task is to extract GENERAL and ABSTRACT strategies that led to success.

The strategies must be applicable to a BROAD CLASS of differential equation problems.

IMPORTANT RULES (MUST FOLLOW):
- Output ONLY the structured memories.
- Do NOT include concrete numbers, coefficients, or specific equations.
- Do NOT include variable names or symbols from the problem.
- Do NOT include worked examples.
- Strategies must be fully abstract and reusable.
- The DESCRIPTION field must be short and embedding-friendly: it should summarize the structural type of the differential equation and the solving idea using plain natural language only.
- The DESCRIPTION field should mention only equation family, operator pattern, and method family; no formulas, no LaTeX, no symbols.
- Use EXACTLY the format shown below.

PROBLEM (FOR CONTEXT ONLY — DO NOT MENTION):
{question}

MODEL SOLUTION (FOR CONTEXT ONLY — DO NOT MENTION):
{reasoning}

OUTPUT FORMAT (COPY EXACTLY):

MEMORY 1:
TITLE: <abstract strategy name>
DESCRIPTION: <short structural signature for retrieval>
CONTENT: <detailed, abstract, reusable reasoning strategy>

MEMORY 2:
TITLE: <abstract strategy name>
DESCRIPTION: <short structural signature for retrieval>
CONTENT: <detailed, abstract, reusable reasoning strategy>

BEGIN GENERATING MEMORIES:
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
- The DESCRIPTION field must be short and embedding-friendly: it should summarize the structural type of the differential equation and the failure/check point using plain natural language only.
- The DESCRIPTION field should mention only equation family, operator pattern, and common pitfall; no formulas, no LaTeX, no symbols.
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
DESCRIPTION: <short structural signature for retrieval>
CONTENT: <abstract lesson or preventive strategy>

MEMORY 2:
TITLE: <abstract mistake or check>
DESCRIPTION: <short structural signature for retrieval>
CONTENT: <abstract lesson or preventive strategy>

BEGIN GENERATING MEMORIES:
"""

KEY_CONCEPT_EXTRACTION_PROMPT = """Your task is to extract the essential structural ideas from a differential equation problem.

Do NOT solve the problem.

Your goal is to produce a compact structural signature of the differential equation that can be matched against stored memories using semantic similarity.

IMPORTANT RULES:
- Use only the LaTeX form and the equation structure.
- Focus on the differential equation type, order, linearity, homogeneity, presence of forcing terms, variable coefficients, separability, exactness, special operators, and other structural cues.
- Do NOT mention concrete numbers, coefficients, or specific solutions.
- Do NOT mention variable names or symbols from the problem.
- Do NOT include calculations.
- Keep the output short, precise, and retrieval-oriented.
- Write in plain natural language only.
- No LaTeX in the output.

Output format:
STRUCTURAL SIGNATURE: [one short sentence]
KEY STRUCTURAL FEATURES: [2-5 short phrases describing the equation class and operators]
LIKELY SOLUTION FAMILY: [one short phrase naming the relevant method family]

Problem:
{question}

BEGIN GENERATING INFORMATION ABOUT PROBLEM:
"""

QUESTION_PROMPT = """Solve the following differential equation concisely with essential derivation steps only.

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

QUESTION_PROMPT_WITH_MEMORIES = """Solve the following differential equation step by step.

Problem (in LaTeX):
{question}

If strategies below are relevant for the current problem, use them to improve the solving process.
Ignore non-relevant strategies.

STRATEGIES:
{memories}

INSTRUCTIONS:
1. Solve step by step for y(x).
2. Keep the solution concise and avoid obvious transitions or verbose explanations.
3. Use valid LaTeX notation.
4. The final solution MUST be in the form:

Final answer: \\boxed{{y(x) = <solution>}}

5. Do NOT include anything after the boxed expression.

BEGIN SOLUTION:
"""

JUDGE_PROMPT = """You are a strict mathematical judge.

Your task is to decide whether the model answer is mathematically equivalent to the reference answer.

Rules:
- Compare only the final answers.
- Ignore formatting differences, whitespace, wording, and explanation.
- If the answers are equivalent, output YES.
- Otherwise output NO.
- Output ONLY YES or NO.

Problem:
{question}

Model answer:
{predicted}

Reference answer:
{reference}

VERDICT:
"""
