from __future__ import annotations

from dataclasses import dataclass

from u_trust.core.types import RouteAction


@dataclass(frozen=True)
class TrustRoutingPolicy:
    tau_pass: float = 0.30
    tau_quarantine: float = 0.70

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_pass <= self.tau_quarantine <= 1.0):
            raise ValueError("Require 0 <= tau_pass <= tau_quarantine <= 1")

    def route(self, node_risk: float) -> RouteAction:
        if node_risk < self.tau_pass:
            return RouteAction.PASS
        if node_risk < self.tau_quarantine:
            return RouteAction.VERIFY
        return RouteAction.QUARANTINE
