from __future__ import annotations

from collections.abc import Iterable


def effective_edge_risk(edge_risk: float, sender_previous_risk: float, eta: float) -> float:
    for name, x in (("edge_risk", edge_risk), ("sender_previous_risk", sender_previous_risk), ("eta", eta)):
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"{name} must be in [0,1], got {x}")
    return float(edge_risk * (eta + (1.0 - eta) * sender_previous_risk))


def noisy_or_node_risk(incoming: Iterable[tuple[float, float]], eta: float) -> float:
    survival = 1.0
    seen = False
    for edge_risk, sender_risk in incoming:
        seen = True
        survival *= 1.0 - effective_edge_risk(edge_risk, sender_risk, eta)
    return float(1.0 - survival) if seen else 0.0


class RiskState:
    def __init__(self, priors: dict[str, float] | None = None):
        self._risk = dict(priors or {})

    def get(self, agent: str) -> float:
        return float(self._risk.get(agent, 0.0))

    def set(self, agent: str, risk: float) -> None:
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be in [0,1]")
        self._risk[agent] = float(risk)

    def snapshot(self) -> dict[str, float]:
        return dict(self._risk)
