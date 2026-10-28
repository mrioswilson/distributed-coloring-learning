#!/usr/bin/env python3
"""Re-execute the first failure in all fifteen frozen confirmatory cells.

The original complete inference batch is preserved; outputs go into closure
artifacts. These are deterministic replays, not additional statistical runs.
"""
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

from dcl.study.storage import atomic_json, file_hash

BASE = Path("results/multiteacher-degree-generalization")
OUT = Path("results/final-closure")


def worker(job):
    import torch
    torch.set_num_threads(1)
    cp, condition, plan = job
    assert file_hash(cp["path"]) == cp["sha256"]
    spec = importlib.util.spec_from_file_location("preserved_replay", "scripts/diagnose-degree-failure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = OUT / "failure-replays" / cp["id"] / (condition["id"] + ".json")
    # Redirect the diagnostic output; input ROOT and all computations stay frozen.
    module.atomic_json = lambda ignored, value: atomic_json(destination, value)
    module.replay(cp, condition, plan)
    r = json.loads(destination.read_text())
    assert r["replay_matches_original"]
    frames = r["frames"]
    first_action = next((f for f in frames if f["predicted_color"] != f["exact_color"] or
                         f["predicted_mark"] != f["exact_mark"]), None)
    first_write = next((f for f in frames if f["forbidden_writes"]), None)
    return dict(checkpoint=cp["id"], sha256=cp["sha256"], teacher=cp["teacher"], architecture=cp["architecture"],
                method=cp["method"], seed=cp["seed"], primary=cp["primary"], condition=condition["id"],
                n=condition["n"], delta=condition["delta"], palette=condition["palette"],
                case_id=r["case"]["case_id"], family=r["case"]["family"], graph_seed=r["case"]["graph_seed"],
                tape_seed=r["case"]["tape_seed"], failure=r["case"]["failure"],
                first_action_step=first_action["step"] if first_action else None,
                first_action_phase=first_action["phase"] if first_action else None,
                first_action_components=first_action["component_disagreements"] if first_action else None,
                first_forbidden_write_step=first_write["step"] if first_write else None,
                forbidden_vertices=first_write["forbidden_writes"] if first_write else [],
                frames=len(frames), sampler=r["sampler"], replay_matches_original=r["replay_matches_original"],
                path=str(destination), replay_sha256=file_hash(destination))


def main():
    tick = time.perf_counter()
    inv = json.loads((BASE / "checkpoint-inventory.json").read_text()) + json.loads((BASE / "supplemental-checkpoint-inventory.json").read_text())
    weights = {c["id"]: c for c in inv}
    plan = json.loads((BASE / "confirmatory/design.json").read_text())
    cells = json.loads((BASE / "confirmatory/index.json").read_text())
    jobs = [(weights[c["checkpoint_id"]], c["condition"], plan) for c in cells if c["successes"] < c["executions"]]
    assert len(jobs) == 15
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(worker, jobs))
    atomic_json(OUT / "failure-replay-audit.json", dict(rows=rows, replays=len(rows), all_match=all(r["replay_matches_original"] for r in rows),
                seconds=time.perf_counter()-tick, workers=3, script_sha256=file_hash(__file__), new_training_updates=0,
                scope="Replayed first failing case in every failed cell, preserving original batch. First temporal discrepancy is not automatically a sufficient cause. Availability alpha threshold is a diagnostic, not a supervised semantic bit."))


if __name__ == "__main__":
    main()
