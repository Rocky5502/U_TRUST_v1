from __future__ import annotations

import argparse
import pandas as pd
from u_trust.risk.calibrator import LogisticRiskCalibrator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    required = {"H", "D", "C", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if "split" in df.columns and (df["split"].astype(str).str.lower() == "test").any():
        raise ValueError("Refusing to fit calibrator on a file containing test rows")
    cal = LogisticRiskCalibrator.fit(df[["H", "D", "C"]].to_numpy(), df["label"].to_numpy(), seed=args.seed)
    cal.save(args.output)
    print("Saved development-fitted calibrator to", args.output)


if __name__ == "__main__":
    main()
