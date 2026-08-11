from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from u_trust.core.types import EpisodeRecord


def append_episode_jsonl(path: str | Path, record: EpisodeRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
