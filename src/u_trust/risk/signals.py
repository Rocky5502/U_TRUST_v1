from __future__ import annotations

from pathlib import Path

from u_trust.backends.base import ChoiceScoringBackend
from u_trust.core.types import EdgeSignals, MessageEnvelope
from u_trust.risk.math import js_divergence, normalized_entropy, total_variation

LEGITIMACY_CHOICES = ("AUTHORIZED", "AMBIGUOUS", "CONFLICTING")
ACTION_CHOICES = ("PASS", "VERIFY", "QUARANTINE")


def _load_prompt(name: str) -> str:
    root = Path(__file__).resolve().parents[3]
    return (root / "prompts" / name).read_text(encoding="utf-8")


def legitimacy_distribution(backend: ChoiceScoringBackend, env: MessageEnvelope) -> dict[str, float]:
    prompt = _load_prompt("legitimacy.txt").format(
        goal=env.goal,
        receiver_role=env.receiver_role,
        provenance=env.provenance,
        message=env.message,
    )
    return backend.score_choices(prompt, LEGITIMACY_CHOICES)


def action_distribution(backend: ChoiceScoringBackend, env: MessageEnvelope, include_message: bool) -> dict[str, float]:
    block = f"Incoming candidate message:\n{env.message}" if include_message else "No new inter-agent message has been introduced."
    prompt = _load_prompt("action_counterfactual.txt").format(
        goal=env.goal,
        receiver_role=env.receiver_role,
        approved_history=env.approved_history or "(empty)",
        message_block=block,
    )
    return backend.score_choices(prompt, ACTION_CHOICES)


def independent_distribution(backend: ChoiceScoringBackend, env: MessageEnvelope) -> dict[str, float]:
    prompt = _load_prompt("independent_view.txt").format(
        goal=env.goal,
        receiver_role=env.receiver_role,
        provenance=env.provenance,
        message=env.message,
    )
    return backend.score_choices(prompt, LEGITIMACY_CHOICES)


def compute_edge_signals(backend: ChoiceScoringBackend, env: MessageEnvelope) -> EdgeSignals:
    local = legitimacy_distribution(backend, env)
    without = action_distribution(backend, env, include_message=False)
    with_message = action_distribution(backend, env, include_message=True)
    independent = independent_distribution(backend, env)
    return EdgeSignals(
        entropy_h=normalized_entropy(local),
        divergence_d=js_divergence(without, with_message),
        disagreement_c=total_variation(local, independent),
        legitimacy_probs=local,
        action_probs_without=without,
        action_probs_with=with_message,
        independent_probs=independent,
    )
