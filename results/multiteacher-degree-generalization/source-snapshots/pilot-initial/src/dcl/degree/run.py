"""Reproducible CPU-only frozen inference campaign; no training entry point."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import torch

from dcl.samplecomplexity.model import QueryPolicy
from dcl.study.storage import atomic_json, file_hash
from .adapter import FrozenAdapter
from .evaluation import evaluate
from .graphs import FAMILIES, make_case
from .inventory import ROOT


def freeze(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"Refusing to change frozen artifact: {path}")
    else:
        atomic_json(path, payload)


def source_hashes():
    paths = list(Path("src/dcl/degree").glob("*.py"))
    paths += [Path("src/dcl/multiteacher/programs.py"), Path("src/dcl/multiteacher_e1/model.py"),
              Path("src/dcl/integration/models.py"), Path("src/dcl/samplecomplexity/model.py"),
              Path("src/dcl/paperstudy/models.py"), Path("src/dcl/universal/model.py")]
    return {str(p): file_hash(p) for p in paths}


def suite_for(plan, condition, replica):
    return [make_case(plan["namespace"], condition["n"], condition["delta"],
                      condition["families"][i % len(condition["families"])],
                      i, replica) for i in range(condition["runs"])]


def worker(job):
    stage, cp, plan, conditions = job
    torch.set_num_threads(1)
    model = QueryPolicy(cp["teacher"], cp["architecture"])
    if file_hash(cp["path"]) != cp["sha256"]:
        raise RuntimeError("Frozen weights changed")
    model.load_state_dict(torch.load(cp["path"], map_location="cpu", weights_only=False)["model"])
    adapter = FrozenAdapter(model)
    replica = cp["seed"] % 100 - 1
    reports = []
    for condition in conditions:
        out = ROOT / stage / cp["id"] / condition["id"]
        done = out / "summary.json"
        if done.exists():
            data = json.loads(done.read_text())
            if file_hash(out / "rows.json") != data["rows_sha256"]:
                raise RuntimeError("Existing outcomes changed")
            reports.append(data)
            continue
        cases = suite_for(plan, condition, replica)
        rows, traces, cost = evaluate(adapter, cp["teacher"], cases, condition["palette"],
                                      batch=plan["batch"], diagnostics=True, trace_limit=1)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "rows.json", rows)
        if traces:
            atomic_json(out / "first-failure.json", traces)
        report = dict(checkpoint_id=cp["id"], sha256=cp["sha256"], teacher=cp["teacher"],
                      architecture=cp["architecture"], method=cp["method"], cohort=cp["cohort"],
                      primary=cp["primary"], seed=cp["seed"], condition=condition,
                      executions=len(rows), successes=sum(r["success"] for r in rows), cost=cost,
                      rows_sha256=file_hash(out / "rows.json"), rows_path=str(out / "rows.json"),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        atomic_json(done, report)
        reports.append(report)
    return reports


def pilot_conditions():
    conditions = []
    for d in (2, 3, 4, 5, 10, 20, 50, 100, 200):
        conditions.append(dict(id=f"global-n256-d{d}-regular", n=256, delta=d, palette="global", families=["regular"], runs=1))
    for n in (64, 128, 512, 1024):
        for d in (3, 10):
            conditions.append(dict(id=f"global-n{n}-d{d}-regular", n=n, delta=d, palette="global", families=["regular"], runs=1))
    conditions.append(dict(id="global-n64-d4-families", n=64, delta=4, palette="global", families=list(FAMILIES), runs=10))
    conditions.append(dict(id="local-n64-d4-heterogeneous", n=64, delta=4, palette="local", families=["isolates", "heterogeneous"], runs=2))
    for d in (50, 200):
        conditions.append(dict(id=f"local-n256-d{d}-isolates", n=256, delta=d, palette="local", families=["isolates"], runs=1))
    return conditions


def execute(stage, plan, inventory, workers):
    # Validate every exact reference tape before any student's outcomes are
    # produced. Shared instances are generated from the same frozen identifiers.
    start = time.perf_counter()
    validated = []
    active_teachers = sorted({cp["teacher"] for cp in inventory})
    for teacher in active_teachers:
        for c in plan["conditions"]:
            for replica in range(5):
                path = ROOT / stage / "exact" / teacher / c["id"] / f"r{replica}.json"
                if path.exists():
                    rows = json.loads(path.read_text())
                else:
                    cases = suite_for(plan, c, replica)
                    rows, _, cost = evaluate(None, teacher, cases, c["palette"], diagnostics=False)
                    atomic_json(path, rows)
                if not all(r["success"] for r in rows):
                    raise RuntimeError(f"Exact teacher failed; student evaluation forbidden: {teacher}/{c['id']}/{replica}")
                validated.append(dict(path=str(path), sha256=file_hash(path), executions=len(rows)))
    atomic_json(ROOT / stage / "exact-validation.json", dict(files=validated, wall_seconds=time.perf_counter()-start))
    jobs = []
    for cp in inventory:
        conds = plan["conditions"]
        if stage == "pilot" and not cp["primary"]:
            ids = {"global-n64-d4-families", "local-n64-d4-heterogeneous", "global-n256-d10-regular"}
            conds = [c for c in conds if c["id"] in ids]
        jobs.append((stage, cp, plan, conds))
    tick = time.perf_counter()
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(worker, jobs):
            reports.extend(result)
            print(json.dumps(dict(stage=stage, checkpoint=result[0]["checkpoint_id"],
                successes=sum(r["successes"] for r in result), runs=sum(r["executions"] for r in result))), flush=True)
    atomic_json(ROOT / stage / "index.json", reports)
    atomic_json(ROOT / stage / "cost.json", dict(wall_seconds=time.perf_counter()-tick,
        cpu_seconds=sum(r["cost"]["cpu_seconds"] for r in reports), workers=workers,
        peak_worker_rss_bytes=max(r["peak_rss_bytes"] for r in reports), gpu_seconds=0,
        new_training_updates=0, max_cuda_memory_bytes=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "confirmatory"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    inventory = json.loads((ROOT / "checkpoint-inventory.json").read_text())
    inventory = [r for r in inventory if r["teacher"] != "BE_BASE_D3"]
    if args.stage == "pilot":
        path = ROOT / "pilot" / "design.json"
        if not path.exists():
            freeze(path, dict(namespace="degree-20261025-pilot-v1", batch=8, conditions=pilot_conditions(),
                checkpoints=[r["id"] for r in inventory], sources=source_hashes(),
                created_utc=datetime.now(timezone.utc).isoformat(), exploratory=True))
        plan = json.loads(path.read_text())
    else:
        path = ROOT / "confirmatory" / "design.json"
        if not path.exists():
            raise RuntimeError("Confirmatory design must be frozen after pilot and before final data")
        plan = json.loads(path.read_text())
        inventory = [r for r in inventory if r["id"] in plan["checkpoints"]]
        if source_hashes() != plan["sources"]:
            raise RuntimeError("Confirmatory source hashes differ from freeze")
    execute(args.stage, plan, inventory, args.workers)


if __name__ == "__main__":
    main()
