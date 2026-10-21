"""Audit the completed, preregistered UCGNN queues directly from saved JSON.

Writes experiment-status.json, run-inventory.csv, evaluation-inventory.csv,
extreme-by-model.csv and extreme-failures.csv. Does not train or alter records.
Counts executions and distinct graph fingerprints separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def audit(root, extreme_root, output):
    repo = Path(__file__).resolve().parents[1]
    issues, metadata_notes, inputs = [], [], hashlib.sha256()
    hashes = {}

    def read(path):
        data = path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        if path.is_relative_to(root):
            identity = "universal/" + path.relative_to(root).as_posix()
        elif extreme_root is not None and path.is_relative_to(extreme_root):
            identity = "extreme/" + path.relative_to(extreme_root).as_posix()
        else:
            identity = path.relative_to(repo).as_posix()
        inputs.update(identity.encode() + b"\0" + checksum.encode() + b"\n")
        return json.loads(data)

    def checkpoint_hash(path):
        if path not in hashes:
            with path.open("rb") as handle:
                hashes[path] = hashlib.file_digest(handle, "sha256").hexdigest()
        return hashes[path]

    plans = {}
    for path in sorted((repo / "configs/ucgnn").glob("*.json")):
        plan = read(path)
        for identity, config in zip(plan["ids"], plan["experiments"], strict=True):
            if identity in plans and plans[identity] != config:
                issues.append(f"Conflicting plans for {identity}")
            plans[identity] = config
    runs, expected_views = [], set()
    for identity, config in sorted(plans.items()):
        directory = root / "runs" / identity
        try:
            actual = read(directory / "config.json")
            history = read(directory / "history.json")
            path = directory / f"epoch-{config['epochs']:03d}.pt"
            complete = (actual == config and [r["epoch"] for r in history] == list(range(1, config["epochs"] + 1))
                        and path.is_file() and (directory / "last.pt").is_file())
        except FileNotFoundError:
            history, complete = [], False
        if not complete:
            issues.append(f"Incomplete training or changed configuration: {identity}")
        runs.append({"run": identity, "experiment": config["name"], "seed": config["seed"],
                     "epochs_completed": len(history), "epochs_planned": config["epochs"], "complete": complete,
                     "checkpoint_sha256": checkpoint_hash(path) if complete else ""})
        if config["epochs"] == 120:
            name = f"{config['name'].lower()}-s{config['seed']:02d}-epoch-120"
            expected_views.update((split, name) for split in ("final_test", "ucgnn_extra"))
        else:
            expected_views.add(("ucgnn_pilot", f"{identity}-epoch-020"))
    expected_views.update(("final_test", f"{name}-s00-best") for name in ("e1", "e2", "e3"))
    expected_views.update(("final_test", f"e1-s00-epoch-{epoch:03d}") for epoch in (10, 20, 40, 60, 80, 100))

    views, observed = [], set()
    original_graphs = defaultdict(set)
    for meta_path in sorted((root / "evaluations").glob("*/*/view.json")):
        split, name = meta_path.parent.parent.name, meta_path.parent.name
        observed.add((split, name))
        meta = read(meta_path)
        declared = set()
        for stage in meta.get("stages", [meta]):
            for degree in stage["degrees"]:
                for n in stage["sizes"]:
                    for family in ("REGULAR", "IRREGULAR"):
                        for index in range(stage["counts"][str(n)]):
                            declared.add((f"{split}/d{degree}/{family}/n{n}/{index}", stage["namespace"]))
        # Early view.json files only retain their last invocation. Reconstruct
        # their full cohort from the frozen protocol, never from observed cases.
        cohorts = {
            "final_test": ((3, 5, 8, 10), {40: 100, 160: 100, 640: 100, 2560: 50, 10240: 25, 40960: 10}, 20261014),
            "ucgnn_extra": ((4, 6, 7, 12, 15), {40: 50, 160: 50, 640: 50, 2560: 20, 10240: 10}, 20261019),
        }
        if split in cohorts:
            degrees, sizes, namespace = cohorts[split]
            expected = {(f"{split}/d{d}/{family}/n{n}/{i}", namespace)
                        for d in degrees for n, count in sizes.items()
                        for family in ("REGULAR", "IRREGULAR") for i in range(count)}
            if declared != expected:
                if "stages" in meta:
                    issues.append(f"Declared stages differ from protocol: {split}/{name}")
                else:
                    metadata_notes.append(f"{split}/{name}: legacy metadata retains only {len(declared)} cases; "
                                          f"coverage checked against all {len(expected)} protocol cases")
        else:
            expected = declared
        ids, fingerprints, counts = set(), set(), Counter()
        for path in sorted(meta_path.parent.glob("*.json")):
            if path.name == "view.json":
                continue
            row = read(path)
            identity = (row["graph_id"], row["namespace"])
            if identity in ids:
                issues.append(f"Duplicated graph/tape in {split}/{name}: {identity}")
            ids.add(identity)
            fingerprints.add(row["fingerprint"])
            original_graphs[split].add(row["fingerprint"])
            counts.update(cases=1, successes=int(row["success"]), unsafe_commits=row["unsafe_commits"],
                          uncolored_nodes=row["uncolored_nodes"], monochromatic_edges=row["monochromatic_edges"])
            if row["checkpoint_sha256"] != meta["checkpoint_sha256"] or row["checkpoint_epoch"] != meta["epoch"]:
                issues.append(f"Checkpoint provenance mismatch: {path}")
        checkpoint = Path(meta["checkpoint"])
        local_checkpoint = root / "runs" / checkpoint.parent.name / checkpoint.name
        if checkpoint_hash(local_checkpoint) != meta["checkpoint_sha256"]:
            issues.append(f"Checkpoint hash mismatch: {local_checkpoint}")
        if ids != expected:
            issues.append(f"Cohort incomplete: {split}/{name}, missing={len(expected-ids)}, extra={len(ids-expected)}")
        preregistered_count = {"final_test": 3080, "ucgnn_extra": 1800}.get(split)
        if preregistered_count is not None and len(ids) != preregistered_count:
            issues.append(f"Wrong protocol count: {split}/{name}")
        views.append({"split": split, "view": name, "experiment": meta["config"]["name"],
                      "seed": meta["config"]["seed"], "checkpoint_epoch": meta["epoch"],
                      "expected_cases": len(expected), "metadata_declared_cases": len(declared),
                      **dict(counts), "distinct_graphs": len(fingerprints),
                      "complete": ids == expected, "checkpoint_sha256": meta["checkpoint_sha256"]})
    issues.extend(f"Missing view: {split}/{name}" for split, name in sorted(expected_views-observed))
    issues.extend(f"Unplanned active view: {split}/{name}" for split, name in sorted(observed-expected_views))
    write_csv(output / "run-inventory.csv", runs)
    write_csv(output / "evaluation-inventory.csv", views)

    totals = {}
    for split in sorted(original_graphs):
        selected = [r for r in views if r["split"] == split]
        totals[split] = {"views": len(selected), "cases": sum(r["cases"] for r in selected),
                         "successes": sum(r["successes"] for r in selected),
                         "distinct_graphs": len(original_graphs[split]),
                         "unsafe_commits": sum(r["unsafe_commits"] for r in selected)}
    extreme, failures = {}, []
    if extreme_root is not None:
        state = read(extreme_root / "status.json")
        task_ids = {p.stem for p in (extreme_root / "tasks").glob("*.json")}
        result_ids, groups, all_fingerprints = set(), defaultdict(list), set()
        for path in sorted((extreme_root / "results").glob("*.json")):
            row = read(path)
            result_ids.add(row["task_id"])
            groups[row["experiment"]].append(row)
            all_fingerprints.add(row["graph"]["fingerprint"])
            if row["status"] not in ("ok", "coloring_failure"):
                issues.append(f"Unmeasured extreme result: {path}")
            if row["weights_sha256_before"] != row["weights_sha256_after"] or not row["same_parameter_objects"]:
                issues.append(f"Weights changed during inference: {path}")
            if row["status"] == "coloring_failure":
                failures.append({"task_id": row["task_id"], "experiment": row["experiment"],
                                 "training_seed": row["training_seed"],
                                 **{k: row["graph"][k] for k in ("degree", "n", "family", "graph_id", "graph_seed", "tape_seed")},
                                 **{k: row[k] for k in ("rounds", "uncolored_nodes", "monochromatic_edges", "unsafe_commits",
                                                       "forbidden_offers", "candidate_correct", "states", "colors_used")}})
        if task_ids != result_ids:
            issues.append(f"Extreme task coverage mismatch: missing={len(task_ids-result_ids)}, extra={len(result_ids-task_ids)}")
        if state["status"] != "complete" or state.get("diagnostics_incomplete") or state.get("pending_resource_cases"):
            issues.append("Extreme coordinator has pending work")
        table = []
        for name, rows in sorted(groups.items()):
            table.append({"experiment": name, "replicas": len({r['training_seed'] for r in rows}),
                          "cases": len(rows), "successes": sum(r["success"] for r in rows),
                          "failures": sum(not r["success"] for r in rows),
                          "distinct_graphs": len({r["graph"]["fingerprint"] for r in rows}),
                          "max_degree": max(r["graph"]["degree"] for r in rows),
                          "max_n": max(r["graph"]["n"] for r in rows),
                          "unsafe_commits": sum(r["unsafe_commits"] for r in rows)})
        write_csv(output / "extreme-by-model.csv", table)
        write_csv(output / "extreme-failures.csv", failures)
        extreme = {"coordinator": state, "models": table, "cases": len(result_ids),
                   "distinct_graphs": len(all_fingerprints), "failures": len(failures),
                   "task_files": len(task_ids), "all_tasks_measured": task_ids == result_ids}
    status = {"generated_utc": datetime.now(timezone.utc).isoformat(), "status": "complete" if not issues else "needs_attention",
              "issues": issues, "metadata_notes": metadata_notes,
              "planned_runs": len(plans), "complete_runs": sum(r["complete"] for r in runs),
              "original_cohorts": totals, "extreme": extreme, "input_manifest_sha256": inputs.hexdigest(),
              "checkpoint_hashes": {str(p): h for p, h in sorted(hashes.items())},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "experiment-status.json").write_text(json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ucgnn"))
    parser.add_argument("--extreme-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = audit(args.root.resolve(), args.extreme_root.resolve() if args.extreme_root else None, args.output.resolve())
    print(json.dumps({k: status[k] for k in ("status", "planned_runs", "complete_runs", "original_cohorts", "issues")}, indent=2))
    if status["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
