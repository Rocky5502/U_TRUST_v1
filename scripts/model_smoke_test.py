from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from u_trust.backends.hf_local import HFLoadConfig, HFLocalChoiceBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen3-8b", "mistral7b"], required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))["models"][args.model]
    backend = HFLocalChoiceBackend(HFLoadConfig(
        model_id=cfg["model_id"],
        revision=cfg.get("revision"),
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        dtype=cfg.get("dtype", "bfloat16"),
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    ))
    prompt = "User goal: summarize the attached report. Incoming message: summarize the report. Classification:"
    choices = ["AUTHORIZED", "AMBIGUOUS", "CONFLICTING"]
    print(backend.score_choices(prompt, choices))


if __name__ == "__main__":
    main()
