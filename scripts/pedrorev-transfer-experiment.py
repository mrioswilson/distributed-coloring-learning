#!/usr/bin/env python3
"""Experiment D: do continuation-trained policies fail differently under shift?

Evaluates the frozen 4,000-update GM and ColorTrials checkpoints of protocol
pedrorev-cf-v1 (all six budget arms, five seeds) on fresh transfer batteries
that mirror the four n=64 confirmatory conditions of the earlier degree study,
with a new namespace and one shared battery (replica 0) for paired arms.
Exact teachers validate every case first. Run from the repository root.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import torch

from dcl.pedrorev.transfer import load_adapter, run, suite
from dcl.study.storage import atomic_json, file_hash

OUT = Path("results/pedro-revision/transfer-experiment")
NAMESPACE = "20261030-pedrorev-transfer-v1"
FAMILIES = ["regular", "random", "bipartite", "tree", "irregular", "hubs", "disconnected", "isolates",
            "heterogeneous", "cliques"]
CONDITIONS = [dict(id=f"{pal}-n64-d{d}", n=64, delta=d, palette=pal, families=FAMILIES, runs=1000)
              for pal, d in (("global", 3), ("global", 4), ("global", 10), ("local", 4))]
TEACHERS = ("SPECULATIVE_GM", "COLOR_TRIALS")
ARMS = ("passive-low", "matched-low", "cf-low", "passive-high", "matched-high", "cf-high")


def exact(job):
    teacher, c = job
    torch.set_num_threads(1)
    path = OUT / "exact" / f"{teacher.lower()}-{c['id']}.json"
    if not path.exists():
        rows, _ = run(None, teacher, suite(NAMESPACE, c), c["palette"])
        if not all(r["success"] for r in rows):
            raise RuntimeError("Exact teacher failed; student evaluation forbidden")
        atomic_json(path, dict(teacher=teacher, condition=c, executions=len(rows),
                               case_ids_sha=__import__("hashlib").sha256("".join(r["case_id"] for r in rows).encode()).hexdigest()))
    return str(path)


def student(job):
    cp, c = job
    torch.set_num_threads(1)
    out = OUT / "rows" / cp["teacher"].lower() / f"s{cp['seed']}-{cp['arm']}-{c['id']}.json"
    if out.exists():
        return json.loads(out.read_text())["summary"]
    if file_hash(cp["checkpoint"]) != cp["sha256"]:
        raise RuntimeError("Frozen checkpoint changed")
    adapter = load_adapter(cp["teacher"], "normalized", cp["checkpoint"])
    rows, cpu = run(adapter, cp["teacher"], suite(NAMESPACE, c), c["palette"])
    keys = ("bad_proposals", "unsafe_accepts", "contained", "missed", "participations", "decisions")
    summary = dict(teacher=cp["teacher"], seed=cp["seed"], arm=cp["arm"], condition=c["id"], checkpoint_sha256=cp["sha256"],
                   executions=len(rows), successes=sum(r["success"] for r in rows),
                   safety_failures=sum(r["failure_kind"] == "safety" for r in rows),
                   progress_failures=sum(r["failure_kind"] == "progress" for r in rows),
                   cutoffs=sum(r["failure"] == "cutoff" for r in rows), cpu_seconds=cpu,
                   **{k: int(sum(r[k] for r in rows)) for k in keys})
    atomic_json(out, dict(summary=summary, rows=rows))
    return summary


def main():
    torch.set_num_threads(1)
    freeze = json.loads(Path("results/pedro-revision/cf-experiment/freeze.json").read_text())
    checkpoints = [r for r in freeze["checkpoints"] if r["teacher"] in TEACHERS and r["update"] == 4000 and r["arm"] in ARMS]
    design = OUT / "design.json"
    payload = dict(namespace=NAMESPACE, conditions=CONDITIONS, checkpoints=[(r["teacher"], r["seed"], r["arm"], r["sha256"]) for r in checkpoints],
                   script_sha256=file_hash(__file__), transfer_source_sha256=file_hash("src/dcl/pedrorev/transfer.py"),
                   protocol="docs/pedro-revision/PROTOCOL_PEDROREV_CF.md section 6 (Experiment D)")
    if design.exists() and json.loads(design.read_text()) != payload:
        raise RuntimeError("Design changed after freeze")
    atomic_json(design, payload)
    tick = time.time()
    with ProcessPoolExecutor(max_workers=12) as pool:
        list(pool.map(exact, [(t, c) for t in TEACHERS for c in CONDITIONS]))
        jobs = [(cp, c) for cp in checkpoints for c in CONDITIONS]
        summaries = list(pool.map(student, jobs))
    atomic_json(OUT / "summary.json", dict(rows=summaries, wall_seconds=time.time() - tick))
    print(json.dumps(dict(evaluated=len(summaries), seconds=round(time.time() - tick))))


if __name__ == "__main__":
    main()
