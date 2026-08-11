from __future__ import annotations

import itertools
from u_trust.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")["experiment"]
    fields = ["models", "benchmarks", "topologies", "defenses", "attacks", "seeds"]
    rows = list(itertools.product(*[cfg[f] for f in fields]))
    print("Configured high-level cells:", len(rows))
    print("Columns:", ", ".join(fields))
    for row in rows[:20]:
        print("  ", row)
    if len(rows) > 20:
        print(f"  ... {len(rows)-20} additional cells")


if __name__ == "__main__":
    main()
