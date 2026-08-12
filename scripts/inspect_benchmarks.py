from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
OUT = ROOT / "results" / "pilot" / "benchmark_inventory.json"

AGENTDOJO_SHA = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
AGENTDOJO_BENCHMARK_VERSION = "v1.2.2"
INJECAGENT_SHA = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"


def git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_json(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        return len(obj)
    raise TypeError(f"Unsupported JSON root in {path}: {type(obj).__name__}")


def inspect_agentdojo() -> dict[str, Any]:
    repo = EXTERNAL / "agentdojo"
    if not repo.exists():
        raise FileNotFoundError("external/agentdojo is missing; run scripts/setup_benchmarks.ps1")
    sha = git_sha(repo)
    if sha != AGENTDOJO_SHA:
        raise RuntimeError(f"AgentDojo SHA mismatch: {sha} != {AGENTDOJO_SHA}")

    # Prefer the editable install, but keeping the pinned checkout's src directory
    # first on sys.path makes the inspected code unambiguously match AGENTDOJO_SHA.
    src = repo / "src"
    sys.path.insert(0, str(src))
    try:
        # IMPORTANT: use AgentDojo's supported suite loader. Importing
        # default_suites.v1.workspace directly can trigger a circular import in
        # version-registration modules at this pinned revision.
        from agentdojo.task_suite.load_suites import get_suite

        workspace_task_suite = get_suite(AGENTDOJO_BENCHMARK_VERSION, "workspace")
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        raise RuntimeError(
            "AgentDojo checkout is pinned but its Python dependencies are not installed. "
            "Run: python -m pip install -e .\\external\\agentdojo\n"
            f"Missing import: {missing}"
        ) from exc

    user_ids = sorted(workspace_task_suite.user_tasks)
    injection_ids = sorted(workspace_task_suite.injection_tasks)
    vectors = workspace_task_suite.get_injection_vector_defaults()
    return {
        "sha": sha,
        "benchmark_version": AGENTDOJO_BENCHMARK_VERSION,
        "suite": "workspace",
        "user_task_count": len(user_ids),
        "injection_task_count": len(injection_ids),
        "injection_vector_count": len(vectors),
        "sample_user_task_ids": user_ids[:10],
        "sample_injection_task_ids": injection_ids[:10],
        "sample_injection_vector_ids": sorted(vectors)[:10],
    }


def inspect_injecagent() -> dict[str, Any]:
    repo = EXTERNAL / "InjecAgent"
    if not repo.exists():
        raise FileNotFoundError("external/InjecAgent is missing; run scripts/setup_benchmarks.ps1")
    sha = git_sha(repo)
    if sha != INJECAGENT_SHA:
        raise RuntimeError(f"InjecAgent SHA mismatch: {sha} != {INJECAGENT_SHA}")

    data = repo / "data"
    user_cases = read_jsonl(data / "user_cases.jsonl")
    attacker_dh = read_jsonl(data / "attacker_cases_dh.jsonl")
    attacker_ds = read_jsonl(data / "attacker_cases_ds.jsonl")

    test_files = [
        "test_cases_dh_base.json",
        "test_cases_dh_enhanced.json",
        "test_cases_ds_base.json",
        "test_cases_ds_enhanced.json",
    ]
    test_counts = {name: count_json(data / name) for name in test_files}

    first = user_cases[0] if user_cases else {}
    return {
        "sha": sha,
        "user_case_count": len(user_cases),
        "attacker_case_dh_count": len(attacker_dh),
        "attacker_case_ds_count": len(attacker_ds),
        "test_case_counts": test_counts,
        "sample_user_case_keys": list(first.keys()),
        "sample_user_tool": first.get("User Tool"),
        "sample_level": first.get("Level"),
    }


def main() -> None:
    inventory = {
        "status": "ENGINEERING_ONLY_NOT_PAPER_EVIDENCE",
        "agentdojo": inspect_agentdojo(),
        "injecagent": inspect_injecagent(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps(inventory, indent=2))
    print(f"\nWrote: {OUT.relative_to(ROOT)}")
    print("This inventory is an engineering sanity check, not paper evidence.")


if __name__ == "__main__":
    main()
