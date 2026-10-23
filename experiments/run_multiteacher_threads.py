"""Run the unchanged comparison with memory shared by independent trainers.

Each trainer owns its model, optimizer and minibatch Generator. Initialization
and global torch RNG restoration are serialized. Training has no random torch
operations, dropout or batch normalization; validation is deterministic.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from threading import RLock

import torch

from dcl.multiteacher import experiment
from dcl.study.storage import atomic_json, file_hash, now


def install_schedule(workers):
    lock = RLock()
    initialize = experiment.initialize
    restore = torch.set_rng_state
    def locked_initialize(*args):
        with lock:
            return initialize(*args)
    def locked_restore(*args, **kwargs):
        with lock:
            return restore(*args, **kwargs)
    experiment.initialize = locked_initialize
    torch.set_rng_state = locked_restore
    experiment.ProcessPoolExecutor = lambda **kwargs: ThreadPoolExecutor(max_workers=workers)


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
    archive = args.root / "sources" / "experiments" / source.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, archive)
    atomic_json(args.root / "execution-threaded.json", {
        "created_utc": now(), "actual_trainers": args.trainers, "actual_processes": 1,
        "configured_worker_limit": config["workers"], "initialization_rng_lock": True,
        "reason": "Independent trainers share the Python/PyTorch memory footprint",
        "driver": "experiments/" + source.name, "driver_sha256": file_hash(source),
        "protocol_unchanged": True, "resume_from_saved_optimizer_and_rng": True})
    install_schedule(args.trainers)
    experiment.run(config, args.root)


if __name__ == "__main__":
    main()
