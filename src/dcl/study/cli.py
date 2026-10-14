from __future__ import annotations
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from .config import Config, digest
from .storage import Registry, atomic_json, file_hash, now, provenance, resources, training_lock


def plans():
    main = [Config(degree=d, seed=s) for d in (3, 5, 8, 10) for s in range(20)]
    # Full pilots are two of the 80 runs, not additional experiments.
    first = [main[0], next(c for c in main if c.degree == 10 and c.seed == 0)]
    main = first + [c for c in main if c not in first]
    controls = [replace(c, method="OWN_ONLY") for c in main if c.degree == 3]
    references = [replace(Config(seed=s), method=method) for s in range(5) for method in ("ORACLE_MATCHED", "TRADITIONAL")]
    budgets = {(g, 3) for g in (200, 800, 1600)} | {(800, t) for t in (1, 3, 9)} | {(200, 12), (400, 6), (800, 3), (1200, 2), (2400, 1)}
    epoch_budgets = [replace(Config(seed=s), graphs=g, tapes=t) for s in range(5) for g, t in sorted(budgets)]
    fixed_updates = [replace(c, updates_per_epoch=256) for c in epoch_budgets]
    return {"phase-a": main, "phase-b": controls, "references": references,
            "references-original-budget": [replace(Config(seed=s), method="TRADITIONAL", tapes=1) for s in range(5)],
            "phase-b-all-degrees": [replace(c, method="OWN_ONLY") for c in main],
            "phase-c-epochs": epoch_budgets, "phase-c-updates": fixed_updates,
            "phase-d": [replace(c, family="MIXED") for c in main]}


def read_plan(path):
    value = json.loads(Path(path).read_text())
    if value["protocol"] != "m6study-v1":
        raise ValueError("Unknown study protocol")
    configs = [Config.from_dict(v) for v in value["experiments"]]
    if len({c.id for c in configs}) != len(configs):
        raise ValueError("Duplicate experiments in plan")
    return configs


class Tee:
    def __init__(self, *handles):
        self.handles = handles
    def write(self, value):
        for handle in self.handles:
            handle.write(value)
            handle.flush()
    def flush(self):
        for handle in self.handles:
            handle.flush()


def run_queue(args):
    from .train import train
    configs = read_plan(args.plan)
    root = Path(args.root).resolve()
    registry = Registry(root)
    for config in configs:
        registry.add(config)
    stop_file = root / "STOP"
    if args.resume:
        stop_file.unlink(missing_ok=True)
    elif stop_file.exists():
        raise RuntimeError("Queue is stopped; use --resume to resume explicitly")
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    if provenance()["git_status"] and not args.allow_dirty:
        raise RuntimeError("Main runs require committed source; --allow-dirty is for technical checks")
    with training_lock(root):
        snapshot = resources(root, device=args.device)
        atomic_json(root / "resources.json", snapshot)
        print(f"queue started {now()} device={args.device} experiments={len(configs)}", flush=True)
        for i, config in enumerate(configs):
            if requested or stop_file.exists():
                print("Clean stop requested; completed epochs preserved", flush=True)
                break
            # Registering all 80 in advance makes pending work visible.
            print(f"[{i + 1}/{len(configs)}] {config.id}", flush=True)
            log_path = logs / f"{config.id}-{time.time_ns()}.log"
            with log_path.open("x") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    status = train(config, root, args.device, lambda: requested or stop_file.exists())
                    if status == "paused":
                        break
                except Exception:
                    detail = traceback.format_exc()
                    previous = next(row for row in registry.rows() if row["id"] == config.id)
                    registry.set(config, "failed", previous["epoch"], previous["updates"], detail)
                    atomic_json(root / "runs" / config.id / f"failure-{time.time_ns()}.json", {"time": now(), "traceback": detail})
                    print(detail, flush=True)
                    # A failed full pilot blocks the rest of Phase A.
                    if i < 2 or not args.continue_on_error:
                        raise


def smoke(args):
    from .train import train
    root = Path(args.root).resolve()
    with training_lock(root):
        for degree in (3, 5, 8, 10):
            config = Config(degree=degree, seed=987654, graphs=4, tapes=2, epochs=3,
                            warmup=1, batch_size=8, val_graphs=2, val_rollouts=1,
                            val_sizes=(40,), val_every=1, max_rounds=8, checkpoint_epochs=(1, 2, 3))
            print("SMOKE", config.id, flush=True)
            train(config, root, args.device)


def freeze(args):
    configs = read_plan(args.plan)
    root = Path(args.root).resolve()
    models = []
    import torch
    for config in configs:
        directory = root / "runs" / config.id
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        if not last["complete"]:
            raise RuntimeError(f"Incomplete experiment: {config.id}")
        choices = [("best", "best.pt")] + [(f"epoch-{epoch:03d}", f"epoch-{epoch:03d}.pt") for epoch in getattr(args, "checkpoint_epochs", [])]
        for policy, filename in choices:
            checkpoint = (directory / filename).resolve()
            selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if selected["config"] != config.to_dict():
                raise RuntimeError("Selected checkpoint configuration differs from the frozen plan")
            models.append({"id": config.id if policy == "best" else f"{config.id}-{policy}",
                           "config": config.to_dict(), "checkpoint": str(checkpoint), "checkpoint_policy": policy,
                           "checkpoint_sha256": file_hash(checkpoint), "selected_epoch": selected["epoch"],
                           "selection_score": selected["best_score"] if policy == "best" else None})
    value = {"protocol": "m6study-v1", "created_utc": now(), "models": models,
             "counts": {"40": 100, "160": 100, "640": 100, "2560": 50, "10240": 25, "40960": 10},
             "families": ["REGULAR", "IRREGULAR"], "max_rounds": 256,
             "test_seed_namespace": 20261014, "selection_uses_final_test": False}
    destination = Path(args.output)
    if destination.exists():
        raise RuntimeError("Frozen test protocol already exists; overwrite refused")
    atomic_json(destination, value)
    print(f"Frozen {len(models)} models; final test is now permitted: {destination}")


def evaluate(args):
    import torch
    from dcl.m4_models import M4B
    from .evaluation import evaluation_cases, run_rollout
    from .train import initialize
    frozen = json.loads(Path(args.freeze).read_text())
    root = Path(args.root).resolve()
    evaluation_id = digest(frozen)
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with training_lock(root):
        resources(root, device=args.device)
        for entry in frozen["models"]:
            config = Config.from_dict(entry["config"])
            checkpoint = Path(entry["checkpoint"])
            if file_hash(checkpoint) != entry["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint changed since freeze")
            model, _, _, _ = initialize(config, args.device)
            saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
            model.load_state_dict(saved["model_state_dict"])
            manifest = json.loads((root / "runs" / config.id / "data_manifest.json").read_text())
            forbidden_graphs = {r["fingerprint"] for split in ("train", "validation") for r in manifest[split]}
            for graph, metadata in evaluation_cases(config, "final_test", tuple(map(int, frozen["counts"])), {int(k): v for k, v in frozen["counts"].items()}, frozen["test_seed_namespace"]):
                if requested or (root / "STOP").exists():
                    print("Evaluation stopped between graphs; partial results preserved")
                    return
                output = root / "evaluations" / evaluation_id / entry["id"] / f"{digest(metadata['graph_id'])[:24]}.json"
                if output.exists():
                    if json.loads(output.read_text())["checkpoint_sha256"] != entry["checkpoint_sha256"]:
                        raise RuntimeError("Persisted evaluation belongs to a different checkpoint")
                    continue
                if metadata["fingerprint"] in forbidden_graphs:
                    raise RuntimeError("Final graph overlaps training or validation")
                result = run_rollout(model, graph, metadata["tape_seed"], config.num_colors, frozen["max_rounds"])
                result.pop("colors")
                policy = entry["checkpoint_policy"]
                atomic_json(output, {"experiment_id": entry["id"], "source_experiment_id": config.id,
                                     "checkpoint_policy": policy, "pair_id": digest([config.pair_id, policy]), "seed": config.seed,
                                     "method": config.method, "training_family": config.family,
                                     "training_graphs": config.graphs, "training_tapes": config.tapes,
                                     "training_epochs": config.epochs, "updates_per_epoch": config.updates_per_epoch or 0,
                                     "checkpoint_epoch": entry["selected_epoch"], "evaluation_id": evaluation_id,
                                     "checkpoint_sha256": entry["checkpoint_sha256"], **metadata, **result})
                print(f"{config.id} {metadata['graph_id']} success={result['success']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Reproducible supervised M6 study")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output", default="configs/m6study")
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--root", default="data/m6study")
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")
    for command in ("status", "stop", "aggregate"):
        p = sub.add_parser(command)
        p.add_argument("--root", default="data/m6study")
    check = sub.add_parser("smoke")
    check.add_argument("--root", default="data/m6study-smoke")
    check.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    f = sub.add_parser("freeze")
    f.add_argument("--plan", required=True)
    f.add_argument("--root", default="data/m6study")
    f.add_argument("--output", default="data/m6study/frozen-final-test.json")
    f.add_argument("--checkpoint-epochs", type=int, nargs="*", default=[])
    ev = sub.add_parser("evaluate")
    ev.add_argument("--freeze", required=True)
    ev.add_argument("--root", default="data/m6study")
    ev.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command == "plan":
        for name, configs in plans().items():
            destination = Path(args.output) / f"{name}.json"
            value = {"protocol": "m6study-v1", "name": name, "experiments": [c.to_dict() for c in configs]}
            if destination.exists() and json.loads(destination.read_text()) != json.loads(json.dumps(value)):
                raise RuntimeError(f"Refusing to overwrite different plan {destination}")
            atomic_json(destination, value)
            print(f"{name}: {len(configs)} configurations")
    elif args.command == "run":
        run_queue(args)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "status":
        registry = Registry(args.root)
        rows = registry.rows()
        print(json.dumps({"counts": dict(Counter(r['status'] for r in rows)), "runs": rows}, indent=2))
    elif args.command == "stop":
        root = Path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "STOP").touch()
        print("Stop requested; current epoch will finish and persist")
    elif args.command == "freeze":
        freeze(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "aggregate":
        from .report import aggregate
        aggregate(Path(args.root))


if __name__ == "__main__":
    main()
