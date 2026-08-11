from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class HFLoadConfig:
    model_id: str
    revision: str | None = None
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    enable_thinking: bool = False


class HFLocalChoiceBackend:
    """Sequence-log-likelihood scorer over a small fixed choice set.

    The backend uses the model's chat template, then scores fixed continuations.
    This gives observable class/action probabilities without reading hidden
    chain-of-thought. Qwen3 thinking is disabled in the primary configuration.
    """

    def __init__(self, config: HFLoadConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError('Install local extras: pip install -e ".[local]"') from exc

        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
        )
        kwargs = {
            "revision": config.revision,
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
        }
        compute_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        kwargs["torch_dtype"] = compute_dtype
        if config.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
        self.model.eval()

    def _render_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Return only the requested constrained classification or routing label. "
                    "Do not explain your answer."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "qwen3" in self.config.model_id.lower():
            kwargs["enable_thinking"] = self.config.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _sequence_logprob(self, prompt: str, continuation: str) -> float:
        torch = self.torch
        rendered = self._render_prompt(prompt)
        prompt_ids = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
        continuation_ids = self.tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"]
        device = next(self.model.parameters()).device
        full_ids = torch.cat([prompt_ids, continuation_ids], dim=1).to(device)
        prompt_len = prompt_ids.shape[1]
        with torch.inference_mode():
            logits = self.model(full_ids).logits[:, :-1, :]
        targets = full_ids[:, 1:]
        logp = torch.log_softmax(logits.float(), dim=-1)
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = max(prompt_len - 1, 0)
        return float(token_logp[:, start:].sum().item())

    def score_choices(self, prompt: str, choices: Sequence[str]) -> dict[str, float]:
        scores = np.array([self._sequence_logprob(prompt, str(choice)) for choice in choices], dtype=float)
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        return {str(choice): float(p) for choice, p in zip(choices, probs, strict=True)}
