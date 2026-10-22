"""Paired native costs on successful E1 cases; a post-training descriptive audit."""
from collections import defaultdict
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def supplement(root, output):
    root, output = Path(root), Path(output)
    if json.loads((root / "complete.json").read_text())["status"] != "complete":
        raise RuntimeError("Cost analysis requires the completed frozen evaluation")
    rows = json.loads((root / "evaluation-rows.json").read_text())
    exact = {(r["teacher"], r["id"]): r for r in rows if r["seed"] is None}
    groups = defaultdict(list)
    for row in rows:
        if row["seed"] is not None:
            groups[row["teacher"], row["seed"], row["family"], row["n"]].append(row)
    costs = []
    for (teacher, seed, family, n), cases in groups.items():
        successes = [r for r in cases if r["success"]]
        refs = [exact[teacher, r["id"]] for r in successes]
        if any(not r["success"] for r in refs):
            raise RuntimeError("Paired exact reference failed")
        for learned, reference in zip(successes, refs):
            if learned["fingerprint"] != reference["fingerprint"] or learned["tape_seed"] != reference["tape_seed"]:
                raise RuntimeError("Native costs do not refer to common cases")
        mean = lambda values: sum(values) / len(values) if values else None
        gnn = mean([r["native_steps"] for r in successes])
        oracle = mean([r["native_steps"] for r in refs])
        record = {"teacher": teacher, "seed": seed, "family": family, "n": n,
                  "all_cases": len(cases), "paired_successes": len(successes),
                  "gnn_native_steps": gnn, "exact_native_steps_same_cases": oracle}
        if teacher == "SPECULATIVE_GM":
            fixed = 2 * ((n + 3) // 4)
            record["gnn_repair_nodes"] = gnn - fixed if gnn is not None else None
            record["exact_repair_nodes_same_cases"] = oracle - fixed if oracle is not None else None
        costs.append(record)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "paired-cost.csv"
    fields = list(dict.fromkeys(k for r in costs for k in r))
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(costs)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "rows": len(costs),
                "source_sha256": sha(__file__), "evaluation_rows_sha256": sha(root / "evaluation-rows.json"),
                "frozen_sha256": sha(root / "frozen.json"), "paired_cost_sha256": sha(target),
                "scope": "Within-teacher native steps, paired on learned successes only; no common LOCAL/wall-time unit across algorithms."}
    (output / "paired-cost-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(supplement(args.root, args.output)))
