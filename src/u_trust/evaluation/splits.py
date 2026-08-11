from __future__ import annotations

import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split


def deterministic_group_split(df: pd.DataFrame, group_col: str = "task_id", stratify_col: str | None = "benchmark_attack", dev_fraction: float = 0.30, seed: int = 17) -> pd.DataFrame:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    groups = df[[group_col] + ([stratify_col] if stratify_col else [])].drop_duplicates(group_col)
    stratify = groups[stratify_col] if stratify_col and groups[stratify_col].nunique() > 1 else None
    dev, test = train_test_split(groups, test_size=1.0 - dev_fraction, random_state=seed, stratify=stratify)
    split_map = {g: "dev" for g in dev[group_col]}
    split_map.update({g: "test" for g in test[group_col]})
    out = df.copy()
    out["split"] = out[group_col].map(split_map)
    return out


def manifest_hash(df: pd.DataFrame, columns: list[str]) -> str:
    stable = df[columns].sort_values(columns).to_csv(index=False).encode()
    return hashlib.sha256(stable).hexdigest()
