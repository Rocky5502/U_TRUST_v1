from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "results" / "pilot"
STATUS = "ENGINEERING_ONLY_NOT_PAPER_EVIDENCE"
ROUTES = {"pass", "verify", "quarantine"}
REQUIRED = {
    "status",
    "pilot_kind",
    "official_benchmark_performance",
    "run_id",
    "episode_index",
    "benchmark",
    "task_id",
    "model",
    "seed",
    "topology",
    "attacked",
    "step",
    "sender",
    "receiver",
    "message_sha256",
    "latency_s",
    "H",
    "D",
    "C",
    "u_edge",
    "q_sender_previous",
    "q_receiver",
    "route",
    "legitimacy_probs",
    "action_probs_without",
    "action_probs_with",
    "independent_probs",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{lineno}: expected JSON object")
            row["__lineno"] = lineno
            rows.append(row)
    return rows


def finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def check_prob_dist(name: str, probs: Any, where: str, errors: list[str]) -> None:
    if not isinstance(probs, dict) or not probs:
        errors.append(f"{where}: {name} must be a non-empty object")
        return
    total = 0.0
    for k, v in probs.items():
        if not finite_number(v):
            errors.append(f"{where}: {name}[{k!r}] is not a finite number")
            continue
        fv = float(v)
        if fv < -1e-9 or fv > 1.0 + 1e-9:
            errors.append(f"{where}: {name}[{k!r}]={fv} outside [0,1]")
        total += fv
    if abs(total - 1.0) > 1e-5:
        errors.append(f"{where}: {name} sums to {total:.8f}, expected 1")


def validate_trace(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    errors: list[str] = []
    warnings: list[str] = []

    if not rows:
        return {"path": str(path.relative_to(ROOT)), "ok": False, "errors": ["empty trace"], "warnings": []}

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[int, int]] = set()

    for row in rows:
        line = row["__lineno"]
        where = f"{path.name}:{line}"
        missing = sorted(REQUIRED - set(row))
        if missing:
            errors.append(f"{where}: missing fields {missing}")
            continue

        if row["status"] != STATUS:
            errors.append(f"{where}: unexpected status {row['status']!r}")
        if row["official_benchmark_performance"] is not False:
            errors.append(f"{where}: engineering pilot incorrectly marked official")
        if row["pilot_kind"] != "signal_propagation_integration":
            errors.append(f"{where}: unexpected pilot_kind {row['pilot_kind']!r}")
        if row["topology"] != "chain4":
            errors.append(f"{where}: expected topology chain4")
        if row["route"] not in ROUTES:
            errors.append(f"{where}: invalid route {row['route']!r}")

        for key in ("H", "D", "C", "u_edge", "q_sender_previous", "q_receiver"):
            value = row[key]
            if not finite_number(value):
                errors.append(f"{where}: {key} is not finite")
            elif float(value) < -1e-9 or float(value) > 1.0 + 1e-9:
                errors.append(f"{where}: {key}={value} outside [0,1]")

        if not finite_number(row["latency_s"]) or float(row["latency_s"]) < 0:
            errors.append(f"{where}: invalid latency_s={row['latency_s']!r}")

        for key in ("legitimacy_probs", "action_probs_without", "action_probs_with", "independent_probs"):
            check_prob_dist(key, row[key], where, errors)

        if not isinstance(row["message_sha256"], str) or len(row["message_sha256"]) != 64:
            errors.append(f"{where}: message_sha256 must be 64 hex characters")

        try:
            epi = int(row["episode_index"])
            step = int(row["step"])
        except (TypeError, ValueError):
            errors.append(f"{where}: episode_index/step must be integers")
            continue
        key = (epi, step)
        if key in seen_keys:
            errors.append(f"{where}: duplicate episode/step {key}")
        seen_keys.add(key)
        by_episode[epi].append(row)

    expected_chain = [
        ("external_tool", "worker"),
        ("worker", "verifier"),
        ("verifier", "executor"),
    ]

    for epi, erows in sorted(by_episode.items()):
        erows = sorted(erows, key=lambda r: int(r["step"]))
        steps = [int(r["step"]) for r in erows]
        if steps != [1, 2, 3]:
            errors.append(f"{path.name}: episode {epi} has steps {steps}, expected [1,2,3]")
        pairs = [(r["sender"], r["receiver"]) for r in erows]
        if pairs != expected_chain:
            errors.append(f"{path.name}: episode {epi} chain {pairs}, expected {expected_chain}")
        hashes = {r["message_sha256"] for r in erows}
        if len(hashes) != 1:
            errors.append(f"{path.name}: episode {epi} message hash changes across hops")
        attacked = {bool(r["attacked"]) for r in erows}
        if len(attacked) != 1:
            errors.append(f"{path.name}: episode {epi} attacked flag changes across hops")
        task_ids = {r["task_id"] for r in erows}
        if len(task_ids) != 1:
            errors.append(f"{path.name}: episode {epi} task_id changes across hops")

    run_ids = {r.get("run_id") for r in rows}
    if len(run_ids) != 1:
        errors.append(f"{path.name}: multiple run_ids found: {sorted(run_ids)}")

    models = {r.get("model") for r in rows}
    benchmarks = {r.get("benchmark") for r in rows}
    if len(models) != 1 or len(benchmarks) != 1:
        errors.append(f"{path.name}: mixed models or benchmarks in one trace")

    summary_path = path.with_name(path.stem + ".summary.json")
    if not summary_path.exists():
        errors.append(f"{path.name}: missing summary file {summary_path.name}")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("official_benchmark_performance") is not False:
            errors.append(f"{summary_path.name}: not marked engineering-only")
        if int(summary.get("hops", -1)) != len(rows):
            errors.append(f"{summary_path.name}: hops={summary.get('hops')} but trace has {len(rows)} rows")
        counts = Counter(str(r["route"]) for r in rows)
        if summary.get("route_counts", {}) != dict(counts):
            errors.append(f"{summary_path.name}: route_counts do not match trace")

        attacked_q = [float(r["q_receiver"]) for r in rows if r["attacked"]]
        benign_q = [float(r["q_receiver"]) for r in rows if not r["attacked"]]
        if attacked_q and benign_q:
            ma = sum(attacked_q) / len(attacked_q)
            mb = sum(benign_q) / len(benign_q)
            if ma <= mb:
                warnings.append(
                    f"diagnostic only: mean attacked q ({ma:.4f}) <= benign q ({mb:.4f}); "
                    "expected to be possible in tiny pilots with the placeholder calibrator"
                )

    return {
        "path": str(path.relative_to(ROOT)),
        "ok": not errors,
        "rows": len(rows),
        "episodes": len(by_episode),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate engineering-only U-TRUST pilot traces.")
    parser.add_argument("--root", type=Path, default=PILOT_ROOT)
    args = parser.parse_args()

    paths = sorted(p for p in args.root.rglob("*.jsonl") if p.is_file())
    if not paths:
        raise SystemExit(f"No pilot JSONL traces found under {args.root}")

    reports = [validate_trace(p) for p in paths]
    overall_ok = all(r["ok"] for r in reports)
    report = {
        "status": STATUS,
        "validator": "pilot_trace_integrity_v1",
        "trace_count": len(reports),
        "overall_ok": overall_ok,
        "reports": reports,
    }
    out = PILOT_ROOT / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {out.relative_to(ROOT)}")
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
