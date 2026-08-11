from __future__ import annotations

from typing import Protocol, Sequence


class ChoiceScoringBackend(Protocol):
    """Score a fixed set of textual choices and return normalized probabilities."""

    def score_choices(self, prompt: str, choices: Sequence[str]) -> dict[str, float]: ...
