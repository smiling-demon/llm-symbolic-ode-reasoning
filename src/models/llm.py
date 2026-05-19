from __future__ import annotations

from time import perf_counter
from typing import List, Union, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    """
    Wrapper over a causal language model for chat-style generation.

    Provides:
    - HuggingFace model loading
    - Chat template formatting
    - Batched generation
    - Timing and token usage statistics
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initializes the language model and tokenizer.

        Args:
            model_name (str): HuggingFace model identifier.
            device (str): Device for inference ("cuda:0", "cpu", etc.).
            dtype (torch.dtype): Model precision type.
        """
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Builds chat messages for the model.

        Args:
            prompt (str): User prompt.

        Returns:
            List[Dict[str, str]]: Chat-formatted messages.
        """
        return [
            {
                "role": "system",
                "content": (
                    "You are a precise mathematical assistant "
                    "specialized in solving differential equations."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    # =========================================================
    # GENERATION
    # =========================================================

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.5,
        top_p: float = 0.95,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generates responses for one or multiple prompts.

        Args:
            prompts (str | List[str]): Input prompt(s).
            max_new_tokens (int): Maximum generation length.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.

        Returns:
            Dict[str, Any] | List[Dict[str, Any]]:
                - response (str)
                - token_count (int)
                - avg_time (float)
        """
        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]

        texts = [
            self.tokenizer.apply_chat_template(
                self._build_messages(p),
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        input_ids = model_inputs["input_ids"]

        start_time = perf_counter()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        elapsed = perf_counter() - start_time

        generated_only = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]

        responses = self.tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True,
        )

        outputs = [
            {
                "response": r.strip(),
                "token_count": len(self.tokenizer.tokenize(r)),
                "avg_time": elapsed / len(prompts),
            }
            for r in responses
        ]

        return outputs[0] if single else outputs
