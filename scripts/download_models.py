from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/freeze one U-TRUST model at a time.")
    ap.add_argument("--model", choices=["qwen3-8b", "mistral7b"], required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--revision", default=None, help="Optional exact HF commit SHA to freeze.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))["models"][args.model]
    model_id = cfg["model_id"]
    revision = args.revision or cfg.get("revision")
    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=args.cache_dir,
        ignore_patterns=["*.gguf", "original/*"],
    )
    print(f"model={args.model}")
    print(f"repo_id={model_id}")
    print(f"resolved_snapshot={path}")
    print("Record the resolved Hugging Face snapshot commit in configs/models.yaml before the frozen test run.")


if __name__ == "__main__":
    main()
