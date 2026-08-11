# U-TRUST Experiment Protocol

## Research questions

**RQ1 — Propagation and prediction.** How does uncertainty about instruction legitimacy propagate through inter-agent communication under prompt injection, and how predictive are its trajectories of downstream alignment failure across attack families, topologies, and model families?

**RQ2 — Robustness and utility.** Can uncertainty-guided trust routing and quarantine reduce prompt-injection propagation and attack success while preserving benign task utility, relative to no defense, message tagging, and channel-gating baselines?

## Experimental unit

The primary unit is a benchmark episode converted into a four-agent workflow. Each attacked episode receives a matched benign control with the same user goal, tools, model, topology, decoding configuration, and seed except for the injected content.

## Four-agent roles

Chain: `Planner -> Worker -> Verifier -> Executor`.

Star: `Worker -> Coordinator <- Verifier`, with Planner providing the immutable task decomposition and Coordinator issuing the final action.

Agent roles are prompt/context roles, not four simultaneously resident model replicas.

## Threat model

The attacker may control untrusted external content consumed by an agent or text entering a legitimate communication channel. The attacker may not modify the original user goal, model weights, system prompts, U-TRUST controller code, or frozen calibration parameters.

Attack families: direct peer override; indirect tool/content injection; role confusion / priority inversion; self-propagating instruction template.

Only benchmark-authorized sandbox actions are executed. Do not connect attack episodes to real credentials or production side-effectful services.

## Signals and routing

U-TRUST computes normalized legitimacy entropy H, goal-conditioned counterfactual JSD D, and independent-view disagreement C. A development-fitted logistic calibrator maps `[H,D,C]` to edge risk `u`. Node risk is propagated with noisy-OR and routed to PASS / VERIFY / QUARANTINE using development-selected thresholds.

## Baselines

1. No defense.
2. Message/source tagging.
3. Local channel gate without history.
4. U-TRUST.

## RQ1 metrics

AUROC, AUPRC, Brier, ECE, clean FPR, detection lead time, and per-hop risk trajectories.

## RQ2 metrics

Attack success rate, downstream compromise fraction, propagation depth, benign utility, utility under attack, clean false-block rate, intervention rate, forward passes, token overhead, and latency.

## Freeze workflow

1. Pin benchmark revisions.
2. Freeze task manifest and dev/test IDs.
3. Freeze model IDs and exact Hugging Face revisions.
4. Freeze quantization and decoding.
5. Hash prompts.
6. Fit calibrator on development only.
7. Choose eta and routing thresholds on development only.
8. Write the freeze manifest.
9. Run frozen test once.
10. Generate tables/figures from raw records without manual value edits.
