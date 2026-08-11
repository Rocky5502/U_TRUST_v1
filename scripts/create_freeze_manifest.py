from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from u_trust.utils.freeze import basic_environment_manifest, git_sha, save_manifest, sha256


def ext_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/processed/freeze_manifest.json")
    p.add_argument("--status", choices=["DRAFT", "FROZEN"], default="DRAFT")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    prompt_hashes = {p.name: sha256(p) for p in sorted((root / "prompts").glob("*.txt"))}
    payload = {
        "status": args.status,
        "u_trust_repo_commit": git_sha(root),
        "agentdojo_commit": ext_sha(root / "external" / "agentdojo"),
        "injecagent_commit": ext_sha(root / "external" / "InjecAgent"),
        "prompt_sha256": prompt_hashes,
        "environment": basic_environment_manifest(),
        "models_config_sha256": sha256(root / "configs" / "models.yaml"),
        "experiment_config_sha256": sha256(root / "configs" / "experiment.yaml"),
        "thresholds_config_sha256": sha256(root / "configs" / "thresholds.yaml"),
        "warning": "Set status=FROZEN only after all development choices are complete and before opening test outcomes.",
    }
    save_manifest(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
