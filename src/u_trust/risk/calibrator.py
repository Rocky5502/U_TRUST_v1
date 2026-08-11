from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from u_trust.core.types import EdgeSignals


@dataclass
class LogisticRiskCalibrator:
    weights: tuple[float, float, float]
    intercept: float

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, signals: EdgeSignals) -> float:
        x = np.array([signals.entropy_h, signals.divergence_d, signals.disagreement_c], dtype=float)
        return float(self._sigmoid(float(np.dot(np.asarray(self.weights), x) + self.intercept)))

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, seed: int = 17) -> "LogisticRiskCalibrator":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=seed, class_weight="balanced", max_iter=2000)
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return cls(tuple(float(v) for v in model.coef_[0]), float(model.intercept_[0]))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"weights": list(self.weights), "intercept": self.intercept}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRiskCalibrator":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(float(v) for v in obj["weights"]), float(obj["intercept"]))


def development_default() -> LogisticRiskCalibrator:
    """Non-paper placeholder used only for smoke tests before fitting dev data."""
    return LogisticRiskCalibrator(weights=(1.0, 2.0, 1.5), intercept=-1.8)
