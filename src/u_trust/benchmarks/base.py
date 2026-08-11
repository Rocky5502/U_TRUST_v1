from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkCase:
    benchmark: str
    task_id: str
    user_goal: str
    trusted_context: str
    untrusted_content: str
    attack_family: str
    attacked: bool
    metadata: dict


class BenchmarkAdapter(Protocol):
    def iter_cases(self, split: str) -> list[BenchmarkCase]: ...
