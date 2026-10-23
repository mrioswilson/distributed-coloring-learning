"""Operational recovery: execute the frozen protocol with one trainer in RAM.

Only task scheduling changes. Model/data/configuration, RNGs, losses, native
controllers and the prespecified budget remain those of run_multiteacher.py.
The original runner remains the canonical protocol implementation.
"""
import argparse
from concurrent.futures import Future
import json
from pathlib import Path
import shutil

from dcl.multiteacher import experiment
from dcl.study.storage import atomic_json, file_hash, now


class SerialExecutor:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    archive = args.root / "sources" / "experiments" / source.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, archive)
    atomic_json(args.root / "execution.json", {
        "created_utc": now(), "actual_workers": 1, "configured_worker_limit": 3,
        "reason": "Memory pressure in concurrent trainers; local disk avoids virtiofs mmap overhead",
        "driver": "experiments/" + source.name, "driver_sha256": file_hash(source),
        "protocol_unchanged": True, "resume_from_saved_optimizer_and_rng": True})
    experiment.ProcessPoolExecutor = SerialExecutor
    experiment.run(json.loads(args.plan.read_text()), args.root)


if __name__ == "__main__":
    main()
