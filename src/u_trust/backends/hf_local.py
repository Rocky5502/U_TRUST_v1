from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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
    """Local Hugging Face backend used by U-TRUST.

    The primary U-TRUST signals use sequence log-likelihood over fixed choices,
    which exposes only observable output probabilities rather than private
    chain-of-thought. The same loaded model can also perform deterministic chat
    generation for benchmark-native engineering/dev runs, avoiding a separate
    vLLM server on Windows.
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
        kwargs["dtype"] = compute_dtype
        if config.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _chat_template_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"tokenize": False, "add_generation_prompt": True}
        if "qwen3" in self.config.model_id.lower():
            kwargs["enable_thinking"] = self.config.enable_thinking
        return kwargs

    def _render_messages(self, messages: Sequence[Mapping[str, str]]) -> str:
        normalized = [
            {"role": str(message["role"]), "content": str(message.get("content", ""))}
            for message in messages
        ]
        kwargs = self._chat_template_kwargs()
        try:
            return self.tokenizer.apply_chat_template(normalized, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(normalized, **kwargs)
        except Exception:
            # Conservative fallback for chat templates that reject a role such
            # as `system`. This is used only by benchmark engineering/dev runs.
            flattened = "\n\n".join(
                f"[{m['role'].upper()}]\n{m['content']}" for m in normalized
            )
            return flattened + "\n\n[ASSISTANT]\n"

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
        return self._render_messages(messages)

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

    def generate_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 384,
    ) -> str:
        """Generate one deterministic assistant turn with the already loaded model.

        This method exists for benchmark-native development runs. Final paper
        experiments must freeze the model revision, prompt, and generation
        budget before the held-out test campaign.
        """
        torch = self.torch
        rendered = self._render_messages(messages)
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_len = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
