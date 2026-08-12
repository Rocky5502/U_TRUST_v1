from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from u_trust.backends.hf_local import HFLoadConfig, HFLocalChoiceBackend
from u_trust.core.types import MessageEnvelope
from u_trust.risk.calibrator import development_default
from u_trust.routing.controller import UTrustController
from u_trust.routing.policy import TrustRoutingPolicy

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
STATUS = "ENGINEERING_ONLY_NOT_PAPER_EVIDENCE"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def model_config(name: str) -> HFLoadConfig:
    data = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    cfg = data["models"][name]
    return HFLoadConfig(
        model_id=cfg["model_id"],
        revision=cfg.get("revision"),
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        dtype=cfg.get("dtype", "bfloat16"),
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )


def thresholds() -> tuple[float, float, float]:
    data = yaml.safe_load((ROOT / "configs" / "thresholds.yaml").read_text(encoding="utf-8"))["u_trust"]
    return float(data["eta"]), float(data["tau_pass"]), float(data["tau_quarantine"])


def load_pilot_cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs" / "pilot.yaml").read_text(encoding="utf-8"))


def injecagent_cases(total: int, seed: int) -> list[dict[str, Any]]:
    data = EXTERNAL / "InjecAgent" / "data"
    users = read_jsonl(data / "user_cases.jsonl")
    attackers = read_jsonl(data / "attacker_cases_dh.jsonl") + read_jsonl(data / "attacker_cases_ds.jsonl")
    rng = random.Random(seed)
    rng.shuffle(users)
    rng.shuffle(attackers)
    n_attack = total // 2
    n_clean = total - n_attack
    rows: list[dict[str, Any]] = []

    for i in range(n_clean):
        u = users[i % len(users)]
        template = str(u["Tool Response Template"])
        message = template.replace("<Attacker Instruction>", "No additional instruction is present.")
        rows.append({
            "task_id": f"injecagent-clean-{i:03d}",
            "goal": str(u["User Instruction"]),
            "message": message,
            "provenance": f"InjecAgent tool response: {u['User Tool']}",
            "attacked": False,
            "attack_id": None,
            "native_metadata": {"user_tool": u["User Tool"], "level": u.get("Level")},
        })

    for i in range(n_attack):
        u = users[(n_clean + i) % len(users)]
        a = attackers[i % len(attackers)]
        instruction = str(a["Attacker Instruction"])
        template = str(u["Tool Response Template"])
        message = template.replace("<Attacker Instruction>", instruction)
        rows.append({
            "task_id": f"injecagent-attack-{i:03d}",
            "goal": str(u["User Instruction"]),
            "message": message,
            "provenance": f"InjecAgent tool response: {u['User Tool']}",
            "attacked": True,
            "attack_id": str(a.get("Attack Type", "unknown")),
            "native_metadata": {
                "user_tool": u["User Tool"],
                "level": u.get("Level"),
                "attack_type": a.get("Attack Type"),
                "attacker_tools": a.get("Attacker Tools"),
            },
        })

    rng.shuffle(rows)
    return rows


def agentdojo_cases(total: int, seed: int, benchmark_version: str) -> list[dict[str, Any]]:
    repo = EXTERNAL / "agentdojo"
    sys.path.insert(0, str(repo / "src"))
    from agentdojo.task_suite import get_suite  # type: ignore

    suite = get_suite(benchmark_version, "workspace")
    user_ids = sorted(suite.user_tasks)
    injection_ids = sorted(suite.injection_tasks)
    vector_defaults = suite.get_injection_vector_defaults()
    vector_ids = sorted(vector_defaults)
    rng = random.Random(seed)
    rng.shuffle(user_ids)
    rng.shuffle(injection_ids)
    rng.shuffle(vector_ids)

    n_attack = total // 2
    n_clean = total - n_attack
    rows: list[dict[str, Any]] = []

    for i in range(n_clean):
        uid = user_ids[i % len(user_ids)]
        vid = vector_ids[i % len(vector_ids)]
        user_task = suite.user_tasks[uid]
        message = str(vector_defaults[vid])
        rows.append({
            "task_id": f"{uid}__clean__{vid}",
            "goal": str(user_task.PROMPT),
            "message": message,
            "provenance": f"AgentDojo workspace injection vector {vid} (default benign value)",
            "attacked": False,
            "attack_id": None,
            "native_metadata": {"user_task_id": uid, "vector_id": vid},
        })

    for i in range(n_attack):
        uid = user_ids[(n_clean + i) % len(user_ids)]
        iid = injection_ids[i % len(injection_ids)]
        vid = vector_ids[(n_clean + i) % len(vector_ids)]
        user_task = suite.user_tasks[uid]
        injection_task = suite.injection_tasks[iid]
        benign_value = str(vector_defaults[vid])
        message = f"{benign_value}\n\nAdditional instruction from untrusted content:\n{injection_task.GOAL}"
        rows.append({
            "task_id": f"{uid}__{iid}__{vid}",
            "goal": str(user_task.PROMPT),
            "message": message,
            "provenance": f"AgentDojo workspace injection vector {vid}",
            "attacked": True,
            "attack_id": iid,
            "native_metadata": {"user_task_id": uid, "injection_task_id": iid, "vector_id": vid},
        })

    rng.shuffle(rows)
    return rows


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assessment_to_row(assessment: Any) -> dict[str, Any]:
    s = assessment.signals
    return {
        "H": s.entropy_h,
        "D": s.divergence_d,
        "C": s.disagreement_c,
        "u_edge": assessment.edge_risk,
        "q_sender_previous": assessment.metadata.get("sender_previous_risk"),
        "q_receiver": assessment.node_risk,
        "route": assessment.route.value,
        "legitimacy_probs": s.legitimacy_probs,
        "action_probs_without": s.action_probs_without,
        "action_probs_with": s.action_probs_with,
        "independent_probs": s.independent_probs,
    }


def run(args: argparse.Namespace) -> Path:
    cfg = load_pilot_cfg()
    benchmark_version = str(cfg.get("benchmark_version", "v1.2.2"))
    roles = cfg["roles"]
    eta, tau_pass, tau_quarantine = thresholds()

    print(f"[{STATUS}] Loading {args.model} once on the local GPU...")
    backend = HFLocalChoiceBackend(model_config(args.model))

    if args.benchmark == "injecagent":
        cases = injecagent_cases(args.episodes, args.seed)
    else:
        cases = agentdojo_cases(args.episodes, args.seed, benchmark_version)

    out_dir = ROOT / "results" / "pilot" / args.benchmark / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.benchmark}__{args.model}__chain4__seed{args.seed}__n{args.episodes}"
    trace_path = out_dir / f"{run_id}.jsonl"
    summary_path = out_dir / f"{run_id}.summary.json"
    if trace_path.exists() and not args.force:
        raise FileExistsError(f"{trace_path} already exists. Use --force to overwrite engineering pilot output.")
    trace_path.write_text("", encoding="utf-8")

    route_counts: Counter[str] = Counter()
    attacked_risks: list[float] = []
    benign_risks: list[float] = []
    total_hops = 0

    for episode_index, case in enumerate(cases):
        controller = UTrustController(
            backend=backend,
            calibrator=development_default(),
            policy=TrustRoutingPolicy(tau_pass, tau_quarantine),
            eta=eta,
        )
        history = "Original user objective received and immutable."
        print(f"Episode {episode_index + 1}/{len(cases)} | attacked={case['attacked']} | {case['task_id']}")

        for step, role in enumerate(roles, start=1):
            env = MessageEnvelope(
                step=step,
                sender=str(role["sender"]),
                receiver=str(role["receiver"]),
                message=str(case["message"]),
                provenance=str(case["provenance"]),
                goal=str(case["goal"]),
                receiver_role=str(role["receiver_role"]),
                approved_history=history,
                benchmark=args.benchmark,
                task_id=str(case["task_id"]),
                attacked=bool(case["attacked"]),
            )
            t0 = time.perf_counter()
            assessment = controller.assess(env)
            latency_s = time.perf_counter() - t0
            row = {
                "status": STATUS,
                "pilot_kind": "signal_propagation_integration",
                "official_benchmark_performance": False,
                "run_id": run_id,
                "episode_index": episode_index,
                "benchmark": args.benchmark,
                "benchmark_version": benchmark_version if args.benchmark == "agentdojo" else None,
                "task_id": case["task_id"],
                "model": args.model,
                "seed": args.seed,
                "topology": "chain4",
                "attacked": case["attacked"],
                "attack_id": case["attack_id"],
                "step": step,
                "sender": role["sender"],
                "receiver": role["receiver"],
                "message_sha256": sha_text(str(case["message"])),
                "provenance": case["provenance"],
                "latency_s": latency_s,
                "native_metadata": case["native_metadata"],
                **assessment_to_row(assessment),
            }
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            route_counts[row["route"]] += 1
            (attacked_risks if case["attacked"] else benign_risks).append(float(row["q_receiver"]))
            total_hops += 1
            history += f" Step {step}: {role['sender']} -> {role['receiver']} routed {row['route']}."

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    summary = {
        "status": STATUS,
        "pilot_kind": "signal_propagation_integration",
        "official_benchmark_performance": False,
        "warning": "Do not cite these values as AgentDojo/InjecAgent task-success or attack-success results.",
        "run_id": run_id,
        "benchmark": args.benchmark,
        "model": args.model,
        "episodes": len(cases),
        "hops": total_hops,
        "seed": args.seed,
        "route_counts": dict(route_counts),
        "mean_q_attacked": mean(attacked_risks),
        "mean_q_benign": mean(benign_risks),
        "trace": str(trace_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {trace_path.relative_to(ROOT)}")
    print(f"Wrote: {summary_path.relative_to(ROOT)}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an engineering-only U-TRUST signal/propagation pilot.")
    parser.add_argument("--benchmark", choices=("agentdojo", "injecagent"), required=True)
    parser.add_argument("--model", choices=("qwen3-8b", "mistral7b"), required=True)
    parser.add_argument("--episodes", type=int, default=2, help="Total episodes; approximately half benign and half attacked.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("--episodes must be at least 2 so the pilot contains benign and attacked cases.")
    run(args)


if __name__ == "__main__":
    main()
