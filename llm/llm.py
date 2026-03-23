from __future__ import annotations

from typing import List, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

class LLM:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            return []

        texts = [self._format_prompt(prompt) for prompt in prompts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        responses = []
        for i in range(len(prompts)):
            output_ids = generated_ids[i][len(inputs.input_ids[i]):].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            responses.append(response)

        return responses
