"""Memory-bounded schedule for the unchanged native-policy comparison.

At most one teacher pool is active. Independent initialization replicas run in
threads and use the RNG protection from run_multiteacher_threads.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import torch

from dcl.multiteacher import experiment
from dcl.study.storage import atomic_json, file_hash, now
from run_multiteacher_threads import install_schedule


class TeacherGroupedExecutor:
    def __init__(self, workers):
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.teacher = None
        self.group = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=True)

    def submit(self, function, *args):
        # train_worker(config, root, teacher, seed, ...)
        teacher = args[2]
        if self.teacher is not None and teacher != self.teacher:
            for future in self.group:
                future.result()
            self.group = []
        self.teacher = teacher
        future = self.executor.submit(function, *args)
        self.group.append(future)
        return future


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trainers", type=int, default=2)
    args = parser.parse_args()
    if args.trainers < 1 or args.trainers > 3:
        raise ValueError("Use one to three independent trainers")
    config = json.loads(args.plan.read_text())
    torch.set_num_threads(config["threads"])
    args.root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    dependency = source.with_name("run_multiteacher_threads.py")
    for path in (source, dependency):
        target = args.root / "sources" / "experiments" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(args.root / "execution-grouped.json", {
        "created_utc": now(), "actual_trainers": args.trainers, "actual_processes": 1,
        "grouped_by_teacher": True, "configured_worker_limit": config["workers"],
        "reason": "Only one teacher data pool is hot at a time to avoid memory pressure",
        "driver": "experiments/" + source.name, "driver_sha256": file_hash(source),
        "dependencies": {"experiments/" + dependency.name: file_hash(dependency)},
        "protocol_unchanged": True, "resume_from_saved_optimizer_and_rng": True})
    install_schedule(args.trainers)
    experiment.ProcessPoolExecutor = lambda **kwargs: TeacherGroupedExecutor(args.trainers)
    experiment.run(config, args.root)


if __name__ == "__main__":
    main()
