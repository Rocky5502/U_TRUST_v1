from u_trust.core.types import RouteAction
from u_trust.routing.policy import TrustRoutingPolicy


def test_routes():
    p = TrustRoutingPolicy(0.3, 0.7)
    assert p.route(0.1) == RouteAction.PASS
    assert p.route(0.4) == RouteAction.VERIFY
    assert p.route(0.9) == RouteAction.QUARANTINE
