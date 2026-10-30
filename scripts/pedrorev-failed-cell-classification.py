#!/usr/bin/env python3
"""Classify *every* failed run of the 15 failed confirmatory transfer cells.

Re-executes each failed cell's complete 1,000-run battery (original design,
namespace, replica and batch size) with the contract-accounting evaluator of
`dcl.pedrorev.transfer`, checks that the success count reproduces the frozen
result, and records for every failed run whether an unsafe accepted write
occurred (safety) or not (progress), with the last-step pending decisions.
Read-only with respect to weights and frozen rows.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from dcl.degree.run import suite_for
from dcl.pedrorev.transfer import load_adapter, run
from dcl.study.storage import atomic_json, file_hash

ROOT = Path("results/multiteacher-degree-generalization")
OUT = Path("results/pedro-revision/failed-cell-classification.json")


def job(cell):
    torch.set_num_threads(1)
    plan = json.loads((ROOT / "confirmatory" / "design.json").read_text())
    inv = json.loads((ROOT / "checkpoint-inventory.json").read_text())
    inv += json.loads((ROOT / "supplemental-checkpoint-inventory.json").read_text())
    cp = next(r for r in inv if r["id"] == cell["checkpoint"])
    if file_hash(cp["path"]) != cp["sha256"]:
        raise RuntimeError("Weights changed")
    condition = next(c for c in plan["conditions"] if c["id"] == cell["condition"])
    cases = suite_for(plan, condition, cp["seed"] % 100 - 1)
    adapter = load_adapter(cp["teacher"], cp["architecture"], cp["path"])
    rows, cpu = run(adapter, cp["teacher"], cases, condition["palette"], batch=plan["batch"], last_state=True)
    failed = [r for r in rows if not r["success"]]
    out = dict(checkpoint=cell["checkpoint"], teacher=cp["teacher"], architecture=cp["architecture"], seed=cp["seed"],
               method=cp["method"], cohort=cp["cohort"], condition=cell["condition"], frozen_successes=cell["successes"],
               reproduced_successes=sum(r["success"] for r in rows), failed_runs=len(failed),
               kinds=dict(Counter(r["failure_kind"] for r in failed)), failures=dict(Counter(r["failure"] for r in failed)),
               families=dict(Counter(r["family"] for r in failed)),
               failed_case_ids=[r["case_id"] for r in failed],
               unsafe_accepts_total=sum(r["unsafe_accepts"] for r in rows),
               contained_total=sum(r["contained"] for r in rows), missed_total=sum(r["missed"] for r in rows),
               failed_rows=[{k: r[k] for k in ("case_id", "family", "failure", "failure_kind", "conflicts", "incomplete",
                                              "steps", "unsafe_accepts", "contained", "missed", "first_unsafe", "last")}
                            for r in failed], cpu_seconds=cpu)
    print(json.dumps({k: out[k] for k in ("checkpoint", "condition", "frozen_successes", "reproduced_successes", "kinds")}), flush=True)
    return out


def main():
    cells = [c for c in json.loads(Path("results/final-closure/failed-cells.json").read_text()) if c["stage"] == "confirmatory"]
    assert len(cells) == 15
    with ProcessPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(job, cells))
    atomic_json(OUT, dict(cells=out, script_sha256=file_hash(__file__),
                          evaluator_sha256=file_hash("src/dcl/pedrorev/transfer.py"),
                          scope="All failed runs of the 15 failed confirmatory cells; same batteries and batch size."))


if __name__ == "__main__":
    main()
