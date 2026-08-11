import pandas as pd
from u_trust.evaluation.splits import deterministic_group_split


def test_group_split_keeps_pairs_together():
    df = pd.DataFrame({"task_id": [f"t{i}" for i in range(10) for _ in (0, 1)], "attacked": [False, True] * 10, "benchmark_attack": ["a"] * 10 + ["b"] * 10})
    out = deterministic_group_split(df, stratify_col=None, dev_fraction=0.3, seed=17)
    assert out.groupby("task_id")["split"].nunique().max() == 1
