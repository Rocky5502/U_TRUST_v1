# Paper-to-Code Map

| Paper object | Code location |
|---|---|
| Three-way legitimacy distribution | `src/u_trust/risk/signals.py` |
| Normalized entropy H | `src/u_trust/risk/math.py::normalized_entropy` |
| Counterfactual JSD D | `src/u_trust/risk/signals.py::action_distribution` + `js_divergence` |
| Cross-view disagreement C | `src/u_trust/risk/signals.py::independent_distribution` + `total_variation` |
| Logistic calibrator | `src/u_trust/risk/calibrator.py` |
| Noisy-OR node risk | `src/u_trust/risk/propagation.py` |
| PASS / VERIFY / QUARANTINE | `src/u_trust/routing/policy.py` |
| Chain/star topology | `src/u_trust/core/topology.py` |
| RQ1/RQ2 metrics | `src/u_trust/evaluation/metrics.py` |
| Benchmark adapters | `src/u_trust/benchmarks/` |

Scientific constraints: entropy alone is not treated as an attack detector; private chain-of-thought is not inspected; the calibrator and thresholds are development-only choices; test settings are immutable after freeze.
