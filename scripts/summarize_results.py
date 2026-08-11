from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from u_trust.evaluation.metrics import rq2_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="results/processed/summary.json")
    args = p.parse_args()
    df = pd.read_json(args.input, lines=True)
    groups = {"|".join(map(str, keys)): rq2_metrics(g) for keys, g in df.groupby(["benchmark", "model", "defense"], dropna=False)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(groups, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
