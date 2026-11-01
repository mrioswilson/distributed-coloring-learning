#!/usr/bin/env python3
"""Freeze, replay and summarize the explicitly post-test BE component study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from dcl.coauthorintegration.functional import BEComponentPolicy, MODES
from dcl.multicf.protocol import cases
from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.protocol import passed_run
from dcl.samplecomplexity.training import initialize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/integrated-paper-20261031"
RUNTIME = ROOT / "data/paper-revision-20261030"
PROTOCOL = ROOT / "docs/INTEGRATED_FUNCTIONAL_DIAGNOSTIC_PROTOCOL.md"
SEEDS = (105001, 105008, 105010)
ARMS = ("native", "cf")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def freeze():
    target = OUT / "functional-freeze.json"
    if target.exists():
        raise RuntimeError("A diagnostic freeze already exists; verify it rather than replacing it")
    old = json.loads((ROOT / "results/paper-revision-20261030/weights-frozen.json").read_text())
    for path, sha in old["sources"].items():
        assert digest(ROOT / path) == sha, path
    sources = [*sorted((ROOT / "src/dcl").rglob("*.py")), Path(__file__), PROTOCOL,
               ROOT / "configs/paper-revision-20261030.json"]
    weights = []
    for seed in SEEDS:
        for arm in ARMS:
            path = RUNTIME / f"models/be_base_d3/s{seed}/{arm}/final.pt"
            original = next(r for r in old["weights"] if
                            (r["teacher"], r["seed"], r["arm"]) == ("BE_BASE_D3", seed, arm))
            assert digest(path) == original["sha256"]
            weights.append(dict(seed=seed, arm=arm, path=str(path.relative_to(ROOT)), sha256=digest(path)))
    write_json(target, dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        scope="Post-test diagnostic, fixed selected seeds, original final cases, no training",
        seeds=SEEDS, arms=ARMS, modes=MODES, cases_per_model=1000, max_executions=30000,
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources}, weights=weights,
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        batching=dict(cohort=32, inference_batch=1024, torch_threads_per_process=1)))
    print(json.dumps(dict(freeze=str(target), weights=len(weights), sources=len(sources))), flush=True)


def verify_freeze():
    path = OUT / "functional-freeze.json"
    d = json.loads(path.read_text())
    for p, sha in d["sources"].items():
        assert digest(ROOT / p) == sha, p
    for w in d["weights"]:
        assert digest(ROOT / w["path"]) == w["sha256"], w["path"]
    return d, digest(path)


def execute(job):
    seed, arm, mode, limit, protocol_sha = job
    os.chdir(ROOT)
    torch.set_num_threads(1)
    directory = OUT / ("functional-smoke" if limit else "functional")
    target = directory / f"s{seed}-{arm}-{mode}.json.gz"
    if target.exists():
        with gzip.open(target, "rt") as f:
            old = json.load(f)
        assert old["freeze_sha256"] == protocol_sha
        return old["summary"]
    config = json.loads((ROOT / "configs/paper-revision-20261030.json").read_text())
    suite = cases(config, "final", seed)
    if limit:
        suite = suite[:limit]
    checkpoint = RUNTIME / f"models/be_base_d3/s{seed}/{arm}/final.pt"
    model = initialize("BE_BASE_D3", "normalized", seed).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    policy = BEComponentPolicy(model, mode).eval()
    rows, seconds = evaluate(policy, "BE_BASE_D3", suite, batch_size=1024, cohort=32)
    original_path = ROOT / f"results/paper-revision-20261030/prospective/be_base_d3/s{seed}/{arm}-test.json"
    old = json.loads(original_path.read_text())["rows"][:len(rows)]
    assert len(rows) == len(old)
    fields = ("case_id", "colors", "native_steps", "failure", "invariants", "terminal_correct", "technical_error")
    same = all(all(a[k] == b[k] for k in fields) for a, b in zip(rows, old))
    for a, b in zip(rows, old):
        assert a["case_id"] == b["case_id"]
    rescued = [r["case_id"] for r, b in zip(rows, old) if passed_run(r) and not passed_run(b)]
    harmed = [r["case_id"] for r, b in zip(rows, old) if not passed_run(r) and passed_run(b)]
    failures = {}
    for r in rows:
        if not passed_run(r):
            failures[r["failure"]] = failures.get(r["failure"], 0) + 1
    summary = dict(seed=seed, arm=arm, mode=mode, runs=len(rows),
        successes=sum(passed_run(r) for r in rows), original_successes=sum(passed_run(r) for r in old),
        rescued=len(rescued), newly_failed=len(harmed), failures=failures,
        same_terminal_rows_as_original=same, technical_errors=sum(r["technical_error"] is not None for r in rows),
        seconds=seconds, mean_steps=float(np.mean([r["native_steps"] for r in rows])))
    result = dict(summary=summary, freeze_sha256=protocol_sha, checkpoint_sha256=digest(checkpoint),
        original_result_sha256=digest(original_path), rescued_case_ids=rescued, newly_failed_case_ids=harmed,
        diagnostics=policy.diagnostics(), rows=rows)
    directory.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(result, separators=(",", ":"), sort_keys=True).encode()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(gzip.compress(raw, mtime=0))
    tmp.replace(target)
    if mode == "baseline" and not same:
        raise RuntimeError(f"Original baseline mismatch; preserved {target}")
    if summary["technical_errors"]:
        raise RuntimeError(f"Technical exceptions; preserved {target}")
    return summary


def run(workers, smoke=False):
    _, sha = verify_freeze()
    limit = 32 if smoke else 0
    modes = ("baseline", "kw_all") if smoke else MODES
    jobs = [(s, a, m, limit, sha) for s in SEEDS for a in ARMS for m in modes]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(execute, job) for job in jobs]):
            print(json.dumps(future.result()), flush=True)


def summarize():
    _, sha = verify_freeze()
    rows, inputs = [], {}
    for seed in SEEDS:
        for arm in ARMS:
            for mode in MODES:
                p = OUT / "functional" / f"s{seed}-{arm}-{mode}.json.gz"
                with gzip.open(p, "rt") as f:
                    d = json.load(f)
                assert d["freeze_sha256"] == sha
                rows.append(dict(d["summary"], diagnostics=d["diagnostics"]))
                inputs[str(p.relative_to(ROOT))] = digest(p)
    assert all(r["same_terminal_rows_as_original"] for r in rows if r["mode"] == "baseline")
    assert all(r["technical_errors"] == 0 for r in rows)
    result = dict(scope="Selected consumed cases; retrospective component interventions, no population remedy claim",
        rows=rows, executions=sum(r["runs"] for r in rows), freeze_sha256=sha, inputs=inputs,
        independent_replication_unit="Three selected training seeds; arms share first-stage tensors and paired cases",
        source_sha256=digest(__file__))
    write_json(OUT / "functional-summary.json", result)
    print(json.dumps(dict(executions=result["executions"], summaries=[{k:v for k,v in r.items() if k!="diagnostics"} for r in rows])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("freeze", "smoke", "run", "summarize"))
    p.add_argument("--workers", type=int, default=2, choices=(1,2))
    a = p.parse_args()
    if a.stage == "freeze": freeze()
    elif a.stage == "smoke": run(a.workers, True)
    elif a.stage == "run": run(a.workers)
    else: summarize()


if __name__ == "__main__":
    main()
