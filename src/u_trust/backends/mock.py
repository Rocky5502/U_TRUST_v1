from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class MockChoiceBackend:
    """Deterministic non-LLM backend for tests and schema validation only."""

    def score_choices(self, prompt: str, choices: Sequence[str]) -> dict[str, float]:
        lowered = prompt.lower()
        suspicious = any(k in lowered for k in ("ignore previous", "override", "secret payload", "unauthorized"))
        base = np.ones(len(choices), dtype=float)
        for i, choice in enumerate(choices):
            c = choice.lower()
            if suspicious and c in {"conflicting", "quarantine"}:
                base[i] += 6.0
            elif not suspicious and c in {"authorized", "pass"}:
                base[i] += 6.0
            else:
                digest = hashlib.sha256((prompt + choice).encode()).digest()[0]
                base[i] += (digest % 7) / 20.0
        probs = base / base.sum()
        return {str(c): float(p) for c, p in zip(choices, probs, strict=True)}
