"""Unchanged training with the sparse sequential teacher's contexts in RAM."""
import argparse
import json
from pathlib import Path
import shutil
from threading import Lock

import torch

from dcl.multiteacher import experiment
from dcl.multiteacher.data import load_pool
from dcl.study.storage import atomic_json, file_hash, now
from run_multiteacher_threads import install_schedule
from run_multiteacher_grouped import TeacherGroupedExecutor
from rooted_multiteacher_cache import resident_roots, verify_resident_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trainers", type=int, default=2)
    args = parser.parse_args()
    if args.trainers not in (1, 2, 3):
        raise ValueError("Use one to three independent trainers")
    config = json.loads(args.plan.read_text())
    torch.set_num_threads(config["threads"])
    args.root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    dependencies = [source.with_name(name) for name in
                    ("run_multiteacher_threads.py", "run_multiteacher_grouped.py", "rooted_multiteacher_cache.py")]
    for path in [source] + dependencies:
        target = args.root / "sources" / "experiments" / path.name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    execution = {"created_utc": now(), "actual_trainers": args.trainers, "actual_processes": 1,
                 "grouped_by_teacher": True, "configured_worker_limit": config["workers"],
                 "driver": "experiments/" + source.name, "driver_sha256": file_hash(source),
                 "dependencies": {"experiments/" + p.name: file_hash(p) for p in dependencies},
                 "reason": "Greedy has one supervised root per graph snapshot; retain exactly its two-hop contexts in RAM",
                 "protocol_unchanged": True, "resume_from_saved_optimizer_and_rng": True}
    atomic_json(args.root / "execution-cached.json", execution)
    original_worker = experiment.train_worker
    cache, lock = {}, Lock()
    def worker(cfg, root, teacher, seed, train_manifest, validation_manifest, metadata):
        if teacher != "GREEDY":
            return original_worker(cfg, root, teacher, seed, train_manifest, validation_manifest, metadata)
        folder = Path(root) / "pools" / teacher.lower()
        with lock:
            if teacher not in cache:
                base = load_pool(folder / "train", train_manifest)
                packed = resident_roots(base, Path(root) / "caches" / "greedy-roots")
                count = verify_resident_roots(base, packed)
                print(f"resident cache verified: teacher={teacher} roots={count}", flush=True)
                cache[teacher] = packed
                del base
            pool = cache[teacher]
        validation = load_pool(folder / "validation", validation_manifest)
        return experiment.train_one(cfg, root, teacher, seed, pool, validation, metadata)
    install_schedule(args.trainers)
    experiment.train_worker = worker
    experiment.ProcessPoolExecutor = lambda **kwargs: TeacherGroupedExecutor(args.trainers)
    experiment.run(config, args.root)


if __name__ == "__main__":
    main()
