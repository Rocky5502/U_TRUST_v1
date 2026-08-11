from __future__ import annotations

import json
import subprocess
from pathlib import Path

PINNED_COMMIT = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"


def repo_path(project_root: str | Path = ".") -> Path:
    return Path(project_root) / "external" / "agentdojo"


def verify_checkout(project_root: str | Path = ".") -> bool:
    repo = repo_path(project_root)
    if not repo.exists():
        return False
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return sha == PINNED_COMMIT


def benchmark_cli_command(model_id: str, suite: str = "workspace") -> list[str]:
    return ["python", "-m", "agentdojo.scripts.benchmark", "-s", suite, "--model", "openai-compatible", "--model-id", model_id]


def save_native_metadata(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
