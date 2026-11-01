#!/usr/bin/env python3
"""Independently replay the 15 historical failed transfer cells, by weight SHA.

This checks the new complete classification, not the absent 810,000-row
historical delivery. No original inputs/results are rewritten.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from dcl.degree.run import suite_for
from dcl.pedrorev.transfer import load_adapter, run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/integrated-paper-20261031"
WEIGHTS = ROOT / "data/coauthor-integration/checkpoints"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--recovery-map", type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    source = ROOT / "results/pedro-revision/failed-cell-classification.json"
    reported = json.loads(source.read_text())["cells"]
    degree = ROOT / "results/multiteacher-degree-generalization"
    inventory = json.loads((degree / "checkpoint-inventory.json").read_text())
    inventory += json.loads((degree / "supplemental-checkpoint-inventory.json").read_text())
    design = json.loads((degree / "confirmatory/design.json").read_text())
    mapping = json.loads(a.recovery_map.read_text()) if a.recovery_map else {}
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    weights, plans = {}, []
    for cell in reported:
        cp = next(r for r in inventory if r["id"] == cell["checkpoint"])
        target = WEIGHTS / (cp["sha256"] + ".pt")
        if not target.exists():
            candidates = [Path(p) for p in mapping.get(cp["sha256"], [])]
            candidates += [ROOT / cp["path"], Path(cp["path"])]
            found = next((p for p in candidates if p.is_file() and sha(p) == cp["sha256"]), None)
            if found is None:
                raise FileNotFoundError(f"Recover original checkpoint {cp['sha256']}; no substitute permitted")
            shutil.copyfile(found, target)
        assert sha(target) == cp["sha256"]
        weights[str(target.relative_to(ROOT))] = cp["sha256"]
        plans.append((cell, cp, target))
    inputs = {str(source.relative_to(ROOT)): sha(source),
              str((degree / "confirmatory/design.json").relative_to(ROOT)): sha(degree / "confirmatory/design.json"),
              "scripts/integrated-replay-transfer.py": sha(__file__), **weights}
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src/dcl").rglob("*.py"))})
    freeze = OUT / "transfer-replay-freeze.json"
    if freeze.exists():
        assert json.loads(freeze.read_text())["inputs"] == inputs
    else:
        freeze.write_text(json.dumps(dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            scope="Verification of reported historical failure classification; original model hashes and case protocol",
            inputs=inputs), indent=2) + "\n")
    summaries = []
    for cell, cp, checkpoint in plans:
        filename = cp["id"] + "--" + cell["condition"] + ".json.gz"
        path = OUT / "transfer-replay" / filename
        if path.exists():
            with gzip.open(path, "rt") as f: result = json.load(f)
            assert result["freeze_sha256"] == sha(freeze)
        else:
            condition = next(c for c in design["conditions"] if c["id"] == cell["condition"])
            cases = suite_for(design, condition, cp["seed"] % 100 - 1)
            adapter = load_adapter(cp["teacher"], cp["architecture"], checkpoint)
            rows, cpu = run(adapter, cp["teacher"], cases, condition["palette"], batch=design["batch"], last_state=True)
            failed = [r for r in rows if not r["success"]]
            summary = dict(checkpoint=cp["id"], teacher=cp["teacher"], condition=cell["condition"],
                runs=len(rows), successes=sum(r["success"] for r in rows), failed_runs=len(failed),
                kinds=dict(Counter(r["failure_kind"] for r in failed)),
                unsafe_accepts_total=sum(r["unsafe_accepts"] for r in rows),
                contained_total=sum(r["contained"] for r in rows), missed_total=sum(r["missed"] for r in rows), cpu_seconds=cpu)
            matches = dict(successes=summary["successes"] == cell["frozen_successes"] == cell["reproduced_successes"],
                failed_case_ids=[r["case_id"] for r in failed] == cell["failed_case_ids"],
                kinds=summary["kinds"] == cell["kinds"],
                counters=all(summary[k] == cell[k] for k in ("unsafe_accepts_total", "contained_total", "missed_total")))
            result = dict(summary=summary, matches=matches, rows=rows, checkpoint_sha256=sha(checkpoint),
                          freeze_sha256=sha(freeze))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":"), sort_keys=True).encode(), mtime=0))
        summaries.append(dict(result["summary"], matches=result["matches"], artifact=str(path.relative_to(ROOT)), sha256=sha(path)))
        print(json.dumps(summaries[-1]), flush=True)
        if not all(result["matches"].values()):
            raise RuntimeError("Replay mismatch; both outputs preserved for investigation")
    assert all(sha(ROOT / p) == d for p, d in inputs.items())
    summary = dict(scope="15,000 historical-cell replays, not regeneration of all810,000 observations",
        summaries=summaries, runs=sum(r["runs"] for r in summaries),
        failed_observations=sum(r["failed_runs"] for r in summaries), all_match=True,
        freeze_sha256=sha(freeze), inputs_unchanged=True)
    (OUT / "transfer-replay-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
