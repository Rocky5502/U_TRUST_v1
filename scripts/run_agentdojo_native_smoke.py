from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from u_trust.backends.hf_local import HFLoadConfig, HFLocalChoiceBackend
from u_trust.integrations.agentdojo_hf import HFInProcessAgentDojoLLM

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
STATUS = "ENGINEERING_ONLY_NOT_PAPER_EVIDENCE"
AGENTDOJO_SHA = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"


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


def git_sha(path: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def select_injectable_pair(suite: Any, attack: Any, user_task_id: str | None, injection_task_id: str | None):
    if injection_task_id is None:
        injection_task_id = sorted(suite.injection_tasks)[0]
    injection_task = suite.get_injection_task_by_id(injection_task_id)

    if user_task_id is not None:
        user_task = suite.get_user_task_by_id(user_task_id)
        candidates = attack.get_injection_candidates(user_task)
        return user_task, injection_task, candidates

    for uid in sorted(suite.user_tasks):
        user_task = suite.get_user_task_by_id(uid)
        try:
            candidates = attack.get_injection_candidates(user_task)
        except ValueError:
            continue
        if candidates:
            return user_task, injection_task, candidates
    raise RuntimeError("No injectable AgentDojo workspace user task was found.")


def run(args: argparse.Namespace) -> Path:
    repo = EXTERNAL / "agentdojo"
    if not repo.exists():
        raise FileNotFoundError("external/agentdojo is missing; run scripts/setup_benchmarks.ps1")
    sha = git_sha(repo)
    if sha != AGENTDOJO_SHA:
        raise RuntimeError(f"AgentDojo SHA mismatch: {sha} != {AGENTDOJO_SHA}")

    # Prefer the pinned checkout over any globally installed AgentDojo copy.
    sys.path.insert(0, str(repo / "src"))

    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.attacks.baseline_attacks import DirectAttack, IgnorePreviousAttack, InjecAgentAttack
    from agentdojo.task_suite import get_suite

    attack_types = {
        "direct": DirectAttack,
        "ignore_previous": IgnorePreviousAttack,
        "injecagent": InjecAgentAttack,
    }

    print(f"[{STATUS}] Loading {args.model} once for native AgentDojo execution...")
    backend = HFLocalChoiceBackend(model_config(args.model))
    llm = HFInProcessAgentDojoLLM(backend, max_new_tokens=args.max_new_tokens)
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=llm,
            model_id=None,
            defense=None,
            tool_delimiter="tool",
            system_message_name=None,
            system_message=None,
            tool_output_format="json",
        )
    )
    pipeline.name = f"u-trust-dev-{args.model}-native-agentdojo"

    suite = get_suite(args.benchmark_version, "workspace")
    attack = attack_types[args.attack](suite, pipeline)
    user_task, injection_task, candidates = select_injectable_pair(
        suite, attack, args.user_task, args.injection_task
    )

    print(f"User task: {user_task.ID}")
    print(f"Injection task: {injection_task.ID}")
    print(f"Injectable vectors: {candidates}")

    t0 = time.perf_counter()
    clean_utility, clean_aux = suite.run_task_with_pipeline(
        pipeline,
        user_task,
        None,
        {},
    )
    clean_seconds = time.perf_counter() - t0

    injections = attack.attack(user_task, injection_task)
    t1 = time.perf_counter()
    attacked_utility, injection_goal_achieved = suite.run_task_with_pipeline(
        pipeline,
        user_task,
        injection_task,
        injections,
    )
    attacked_seconds = time.perf_counter() - t1

    result = {
        "status": STATUS,
        "native_agentdojo_evaluation": True,
        "paper_evidence": False,
        "warning": "Development smoke only. Do not cite these two task outcomes in the paper.",
        "benchmark": "agentdojo",
        "benchmark_version": args.benchmark_version,
        "benchmark_sha": sha,
        "model": args.model,
        "model_id": backend.config.model_id,
        "model_revision": backend.config.revision,
        "attack": args.attack,
        "user_task_id": user_task.ID,
        "injection_task_id": injection_task.ID,
        "injection_vector_ids": sorted(injections),
        "clean": {
            "utility": bool(clean_utility),
            "auxiliary_security_flag": bool(clean_aux),
            "latency_s": clean_seconds,
        },
        "attacked": {
            "utility": bool(attacked_utility),
            "injection_goal_achieved": bool(injection_goal_achieved),
            "attack_success": bool(injection_goal_achieved),
            "latency_s": attacked_seconds,
        },
    }

    out_dir = ROOT / "results" / "dev_native" / "agentdojo" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"native_smoke__{args.attack}__{user_task.ID}__{injection_task.ID}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out.relative_to(ROOT)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one clean and one attacked native AgentDojo task with an in-process HF model."
    )
    parser.add_argument("--model", choices=("qwen3-8b", "mistral7b"), required=True)
    parser.add_argument("--attack", choices=("direct", "ignore_previous", "injecagent"), default="injecagent")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--user-task", default=None)
    parser.add_argument("--injection-task", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
