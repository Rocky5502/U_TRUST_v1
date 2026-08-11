from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from u_trust.evaluation.splits import deterministic_group_split, manifest_hash


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="results/processed/split_manifest.csv")
    p.add_argument("--group-col", default="task_id")
    p.add_argument("--stratify-col", default="benchmark_attack")
    p.add_argument("--dev-fraction", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    out = deterministic_group_split(df, group_col=args.group_col, stratify_col=args.stratify_col if args.stratify_col in df.columns else None, dev_fraction=args.dev_fraction, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("Wrote", args.output)
    print("Split SHA256:", manifest_hash(out, [args.group_col, "split"]))


if __name__ == "__main__":
    main()
