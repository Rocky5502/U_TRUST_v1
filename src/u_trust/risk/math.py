from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def _vec(p: Mapping[str, float], keys: list[str]) -> np.ndarray:
    arr = np.array([float(p[k]) for k in keys], dtype=float)
    if np.any(arr < 0):
        raise ValueError("Probabilities must be non-negative")
    total = arr.sum()
    if total <= 0:
        raise ValueError("Probability mass must be positive")
    return arr / total


def normalized_entropy(probs: Mapping[str, float]) -> float:
    keys = list(probs)
    p = _vec(probs, keys)
    nz = p[p > 0]
    h = -float(np.sum(nz * np.log(nz)))
    return h / math.log(len(keys)) if len(keys) > 1 else 0.0


def js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    pv = _vec({k: p.get(k, 0.0) for k in keys}, keys)
    qv = _vec({k: q.get(k, 0.0) for k in keys}, keys)
    m = 0.5 * (pv + qv)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return (0.5 * kl(pv, m) + 0.5 * kl(qv, m)) / math.log(2.0)


def total_variation(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    pv = _vec({k: p.get(k, 0.0) for k in keys}, keys)
    qv = _vec({k: q.get(k, 0.0) for k in keys}, keys)
    return 0.5 * float(np.abs(pv - qv).sum())
