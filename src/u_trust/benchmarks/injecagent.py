from __future__ import annotations

import subprocess
from pathlib import Path

PINNED_COMMIT = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"


def repo_path(project_root: str | Path = ".") -> Path:
    return Path(project_root) / "external" / "InjecAgent"


def verify_checkout(project_root: str | Path = ".") -> bool:
    repo = repo_path(project_root)
    if not repo.exists():
        return False
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return sha == PINNED_COMMIT


def integration_note() -> str:
    return "Preserve benchmark-native case IDs and original target/tool metadata; do not edit benchmark labels in-place."
