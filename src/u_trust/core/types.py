from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RouteAction(str, Enum):
    PASS = "pass"
    VERIFY = "verify"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class MessageEnvelope:
    step: int
    sender: str
    receiver: str
    message: str
    provenance: str
    goal: str
    receiver_role: str
    approved_history: str = ""
    benchmark: str | None = None
    task_id: str | None = None
    attacked: bool | None = None


@dataclass(frozen=True)
class EdgeSignals:
    entropy_h: float
    divergence_d: float
    disagreement_c: float
    legitimacy_probs: dict[str, float] = field(default_factory=dict)
    action_probs_without: dict[str, float] = field(default_factory=dict)
    action_probs_with: dict[str, float] = field(default_factory=dict)
    independent_probs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskAssessment:
    edge_risk: float
    node_risk: float
    route: RouteAction
    signals: EdgeSignals
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    episode_id: str
    benchmark: str
    task_id: str
    model: str
    topology: str
    defense: str
    attack_family: str
    attacked: bool
    attack_success: bool
    benign_success: bool | None
    unsafe_action_step: int | None
    detection_step: int | None
    propagation_depth: int
    compromised_agents: int
    total_agents: int
    quarantined_messages: int
    verified_messages: int
    total_messages: int
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
