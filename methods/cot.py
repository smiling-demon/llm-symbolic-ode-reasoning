import json
import torch
import random

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, logging


COT_PROMPT_TEMPLATE = """You are a highly precise reasoning assistant. Solve the problem step by step.

Problem: {question}

INSTRUCTIONS (READ CAREFULLY):
1. Show your step-by-step reasoning in **numbered steps**, e.g.:
   Step 1: ...
   Step 2: ...
   Final answer: ...
3. THE FINAL ANSWER MUST BE STRICTLY AN INTEGER NUMBER. NO UNITS, NO WORDS, NO SYMBOLS, NO LATEX.
4. THE FINAL ANSWER MUST BE OUTPUT IN THIS EXACT FORMAT:
   Final answer: #### <final_number>
   Replace <final_number> with only the INTEGER NUMERIC ANSWER WITHOUT ANITHING ELSE.
5. DO NOT INCLUDE ANYTHING ELSE AFTER '####'.
6. IF YOU CANNOT CALCULATE, OUTPUT ONLY '#### NO ANSWER' AS PLACEHOLDER.

BEGIN STEP-BY-STEP REASONING:
"""


def majority_vote(population):
    answers = []
    for content in population:
        if "####" in content:
            ans = content.split("####")[-1].strip()
            answers.append(ans)

    if not answers:
        return ""

    counter = Counter(answers)
    return counter.most_common(1)[0][0]
    
    
def generate_populations_from_prompts(prompts, num_questions, question_indices, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    texts = [tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    populations = [[] for _ in range(num_questions)]
    for i, q_idx in enumerate(question_indices):
        output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        populations[q_idx].append(content)
    return populations
    

def cot(questions, N=1):
    num_questions = len(questions)

    prompts = []
    question_indices = []
    for q_idx, q in enumerate(questions):
        prompts.append(COT_PROMPT_TEMPLATE.format(question=q))
        question_indices.append(q_idx)

    populations = generate_populations_from_prompts(prompts, num_questions, question_indices)

    final_answers = []
    for pop in populations:
        final_answer = majority_vote(pop)
        final_answers.append(final_answer)
    
    return final_answers
    
