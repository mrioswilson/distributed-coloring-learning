#!/usr/bin/env python3
"""Recompute support/labels and validate published additional-study summaries.

New analysis directory only. Summary validation is not new inference of the
105 unavailable additional-study checkpoints; that distinction is explicit.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/integrated-paper-20261031"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("old_audit", ROOT / "scripts/audit-revision-continuations.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    config = json.loads((ROOT / "configs/paper-revision-20261030.json").read_text())
    inputs, pairs = {}, []
    for teacher in config["teachers"]:
        for seed in config["seeds"]:
            pairs.append(old.inspect_pair(ROOT / "data/paper-revision-20261030", teacher, seed, inputs))
    previous = json.loads((ROOT / "results/paper-revision-20261030/continuation-audit.json").read_text())
    assert pairs == previous["rows"]
    validated = 0
    for name in ("cf-experiment/analysis.json", "cf-followup/analysis.json"):
        path = ROOT / "results/pedro-revision" / name
        d = json.loads(path.read_text())
        for row in d["rows"]:
            for metrics in (row["overall"], *row.get("strata", {}).values()):
                n, tp, tn, fp, fn = (metrics[k] for k in ("n", "tp", "tn", "fp", "fn"))
                assert n == tp + tn + fp + fn
                assert abs(metrics["accuracy"] - (tp + tn) / n) < 1e-12
                ba = .5 * (tp / (tp + fn) + tn / (tn + fp)) if tp + fn and tn + fp else None
                assert metrics["balanced_accuracy"] is None if ba is None else abs(metrics["balanced_accuracy"] - ba) < 1e-12
            validated += 1
        inputs[str(path.relative_to(ROOT))] = sha(path)
    functional = []
    for path in sorted((ROOT / "results/pedro-revision/cf-experiment/functional/be_base_d3").glob("*.json")):
        d = json.loads(path.read_text())
        # These original files contain compact per-case statuses, not just means.
        functional.append(dict(path=str(path.relative_to(ROOT)), **{k:v for k,v in d.items() if k != "rows"}))
        inputs[str(path.relative_to(ROOT))] = sha(path)
    native, cf = {}, {}
    for teacher in config["teachers"]:
        rows = [r for r in pairs if r["teacher"] == teacher]
        native[teacher] = [min(r["logical_test_occurrence_overlap"]["native"] for r in rows), max(r["logical_test_occurrence_overlap"]["native"] for r in rows)]
        cf[teacher] = [min(r["logical_test_occurrence_overlap"]["cf"] for r in rows), max(r["logical_test_occurrence_overlap"]["cf"] for r in rows)]
    result = dict(scope=__doc__, original_20_pair_audit_exactly_reproduced=True,
        raw_labels_and_fine_keys_recomputed=True, native_logical_overlap=native, cf_logical_overlap=cf,
        additional_summary_rows_validated=validated, functional_be_files=len(functional),
        functional_be=functional, pairs=pairs, inputs=inputs, source_sha256=sha(__file__))
    (OUT / "evidence-reconciliation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:result[k] for k in ("original_20_pair_audit_exactly_reproduced","additional_summary_rows_validated","functional_be_files","native_logical_overlap","cf_logical_overlap")}))


if __name__ == "__main__":
    main()
