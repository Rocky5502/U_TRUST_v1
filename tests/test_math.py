import math
from u_trust.risk.math import js_divergence, normalized_entropy, total_variation


def test_entropy_bounds():
    assert math.isclose(normalized_entropy({"a": 1, "b": 0, "c": 0}), 0.0)
    assert math.isclose(normalized_entropy({"a": 1/3, "b": 1/3, "c": 1/3}), 1.0, rel_tol=1e-6)


def test_distances():
    p = {"a": 1.0, "b": 0.0}
    q = {"a": 0.0, "b": 1.0}
    assert 0.99 <= js_divergence(p, q) <= 1.01
    assert math.isclose(total_variation(p, q), 1.0)
