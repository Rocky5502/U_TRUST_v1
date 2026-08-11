from u_trust.risk.propagation import effective_edge_risk, noisy_or_node_risk


def test_effective_risk_increases_with_sender_risk():
    assert effective_edge_risk(0.5, 1.0, eta=0.5) > effective_edge_risk(0.5, 0.0, eta=0.5)


def test_noisy_or_accumulates_parents():
    one = noisy_or_node_risk([(0.4, 0.2)], eta=0.5)
    two = noisy_or_node_risk([(0.4, 0.2), (0.4, 0.2)], eta=0.5)
    assert two > one
    assert 0 <= two <= 1
