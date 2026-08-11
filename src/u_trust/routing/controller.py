from __future__ import annotations

from dataclasses import dataclass

from u_trust.backends.base import ChoiceScoringBackend
from u_trust.core.types import MessageEnvelope, RiskAssessment
from u_trust.risk.calibrator import LogisticRiskCalibrator
from u_trust.risk.propagation import RiskState, noisy_or_node_risk
from u_trust.risk.signals import compute_edge_signals
from u_trust.routing.policy import TrustRoutingPolicy


@dataclass
class UTrustController:
    backend: ChoiceScoringBackend
    calibrator: LogisticRiskCalibrator
    policy: TrustRoutingPolicy
    eta: float = 0.55

    def __post_init__(self) -> None:
        self.state = RiskState()

    def assess(self, env: MessageEnvelope) -> RiskAssessment:
        signals = compute_edge_signals(self.backend, env)
        edge_risk = self.calibrator.predict(signals)
        sender_risk = self.state.get(env.sender)
        node_risk = noisy_or_node_risk([(edge_risk, sender_risk)], eta=self.eta)
        route = self.policy.route(node_risk)
        self.state.set(env.receiver, node_risk)
        return RiskAssessment(
            edge_risk=edge_risk,
            node_risk=node_risk,
            route=route,
            signals=signals,
            metadata={"sender_previous_risk": sender_risk, "eta": self.eta},
        )
