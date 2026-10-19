"""Command line for the Universal Color GNN study (protocol ucgnn-v1)."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import signal
import sys
import time
import traceback

from dcl.study.config import digest
from dcl.study.storage import atomic_json, file_hash, now, provenance, resources, training_lock
from .config import UConfig

DEFAULT_COUNTS = {40: 100, 160: 100, 640: 100, 2560: 50, 10240: 25, 40960: 10}


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


def plans():
    """Preregistered plans (docs/UCGNN_PROTOCOLO.md)."""
    from dataclasses import replace
    e1 = UConfig(name="E1")
    e2 = UConfig(name="E2", degrees=(3, 10), graphs_per_degree=(400, 400), val_graphs_per_degree=100)
    e3 = UConfig(name="E3", degrees=(3, 5, 8, 10), graphs_per_degree=(200,) * 4, val_graphs_per_degree=50)
    return {
        "p0-pilot": [UConfig(name="P0", epochs=20, checkpoint_epochs=(10, 20))],
        "e1": [e1], "e2": [e2], "e3": [e3],
        "a1-raw-prefix": [UConfig(name="A1", variant="RAW_PREFIX")],
        "secondary-full-budget": [replace(e2, name="E2F", graphs_per_degree=(800, 800)),
                                  replace(e3, name="E3F", graphs_per_degree=(800,) * 4)],
        "replication": [replace(c, seed=s) for s in range(1, 5) for c in (e1, e2, e3)],
    }


def write_plans(output):
    for name, configs in plans().items():
        destination = Path(output) / f"{name}.json"
        value = {"protocol": "ucgnn-v1", "name": name, "experiments": [c.to_dict() for c in configs],
                 "ids": [c.id for c in configs]}
        if destination.exists() and json.loads(destination.read_text()) != json.loads(json.dumps(value)):
            raise RuntimeError(f"Refusing to overwrite different plan {destination}")
        atomic_json(destination, value)
        print(f"{name}: {[c.id for c in configs]}")


def read_plan(path):
    value = json.loads(Path(path).read_text())
    if value["protocol"] != "ucgnn-v1":
        raise ValueError("Unknown protocol")
    configs = [UConfig.from_dict(v) for v in value["experiments"]]
    if len({c.id for c in configs}) != len(configs):
        raise ValueError("Duplicate experiments in plan")
    return configs


def stop_handler():
    state = {"requested": False}

    def stop(signum, frame):
        state["requested"] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return state


def run_queue(args):
    from .train import train
    configs = read_plan(args.plan)
    root = Path(args.root).resolve()
    stop_file = root / "STOP"
    if args.resume:
        stop_file.unlink(missing_ok=True)
    elif stop_file.exists():
        raise RuntimeError("Queue is stopped; use --resume")
    if provenance()["git_status"] and not args.allow_dirty:
        raise RuntimeError("Main runs require committed source; --allow-dirty is for technical checks")
    state = stop_handler()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with training_lock(root):
        atomic_json(root / "resources.json", resources(root, device=args.device))
        for i, config in enumerate(configs):
            if state["requested"] or stop_file.exists():
                print("Clean stop requested; completed epochs preserved", flush=True)
                break
            print(f"[{i + 1}/{len(configs)}] {config.id}", flush=True)
            log_path = logs / f"{config.id}-{time.time_ns()}.log"
            with log_path.open("x") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    status = train(config, root, args.device, lambda: state["requested"] or stop_file.exists(),
                                   stop_after_epoch=args.stop_after_epoch)
                    print(f"{config.id} status={status}", flush=True)
                    if status == "paused":
                        break
                except Exception:
                    detail = traceback.format_exc()
                    atomic_json(root / "runs" / config.id / f"failure-{time.time_ns()}.json", {"time": now(), "traceback": detail})
                    print(detail, flush=True)
                    raise


def evaluate(args):
    import torch
    from .evaluation import build_case, cohort_cases, run_rollout
    from .train import load_model
    root = Path(args.root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = file_hash(checkpoint)
    model, saved = load_model(checkpoint, args.device)
    config = UConfig.from_dict(saved["config"])
    run_dir = checkpoint.parent
    manifest = json.loads((run_dir / "data_manifest.json").read_text())
    forbidden = {r["fingerprint"] for split in ("train", "validation") for r in manifest[split]}
    sizes = tuple(args.sizes)
    counts = {n: (args.count if args.count else DEFAULT_COUNTS[n]) for n in sizes}
    view = args.view or f"{config.id}-epoch-{saved['epoch']:03d}"
    out_dir = root / "evaluations" / args.split / view
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(out_dir / "view.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                                        "epoch": saved["epoch"], "config": config.to_dict(), "split": args.split,
                                        "namespace": args.namespace, "degrees": args.degrees, "sizes": list(sizes),
                                        "counts": {str(k): v for k, v in counts.items()}, "created_utc": now(),
                                        "evaluation_provenance": provenance()})
    state = stop_handler()
    cache = root / "cohort-cache"
    with training_lock(root):
        resources(root, device=args.device)
        for degree in args.degrees:
            for metadata in cohort_cases(degree, sizes, counts, split=args.split, namespace=args.namespace):
                if state["requested"] or (root / "STOP").exists():
                    print("Evaluation stopped between graphs; partial results preserved", flush=True)
                    return
                output = out_dir / f"{digest(metadata['graph_id'])[:24]}.json"
                if output.exists():
                    if json.loads(output.read_text())["checkpoint_sha256"] != checkpoint_hash:
                        raise RuntimeError("Persisted evaluation belongs to a different checkpoint")
                    continue
                edges, info = build_case(metadata, cache)
                if info["fingerprint"] in forbidden:
                    raise RuntimeError("Evaluation graph overlaps training or validation")
                result, colors = run_rollout(model, metadata["n"], edges, degree + 1, metadata["tape_seed"], args.max_rounds)
                record = {"view": view, "experiment_id": config.id, "experiment": config.name, "seed": config.seed,
                          "variant": config.variant, "method": config.method, "training_degrees": list(config.degrees),
                          "seen_degree": degree in config.degrees, "checkpoint_epoch": saved["epoch"],
                          "checkpoint_sha256": checkpoint_hash, "parameters": saved["parameters"],
                          **metadata, **info, **result}
                if args.teacher_rounds:
                    from .evaluation import teacher_rollout
                    teacher_colors, teacher_rounds = teacher_rollout(metadata["n"], edges, degree + 1, metadata["tape_seed"])
                    record["teacher_rounds"] = teacher_rounds
                    record["identical_final_coloring"] = bool((teacher_colors == colors).all())
                atomic_json(output, record)
                print(f"{view} {metadata['graph_id']} success={result['success']} rounds={result['rounds']} "
                      f"unsafe={result['unsafe_commits']} t={result['inference_seconds']:.2f}s", flush=True)


def smoke(args):
    """Same model object on Delta=3 and Delta=10, identical parameters, tiny training."""
    import torch
    from dataclasses import replace
    from .evaluation import build_case, cohort_cases, run_rollout
    from .model import UniversalColorGNN, count_parameters
    from .train import train
    root = Path(args.root).resolve()
    model = UniversalColorGNN().to(args.device)
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    print(f"parameters={count_parameters(model)}")
    for degree in (3, 10, 15):
        metadata = next(cohort_cases(degree, (40,), {40: 1}, split="smoke", namespace=1))
        edges, _ = build_case(metadata)
        result, colors = run_rollout(model, 40, edges, degree + 1, metadata["tape_seed"], 16)
        assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == shapes
        print(f"degree={degree} palette={degree + 1} untrained rounds={result['rounds']} states={result['states']} "
              f"max_color={int(colors.max())} parameters={count_parameters(model)}")
    with training_lock(root):
        config = UConfig(name="SMOKE", degrees=(3, 10), graphs_per_degree=(4, 4), tapes=1, epochs=3, warmup=1,
                         batch_size=8, val_graphs_per_degree=2, val_rollouts=1, val_sizes=(40,), val_every=1,
                         max_rounds=16, checkpoint_epochs=(1, 3))
        print(train(config, root, args.device, check_resources=args.device == "cuda"))


def main():
    parser = argparse.ArgumentParser(description="Universal Color GNN study")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output", default="configs/ucgnn")
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--root", default="data/ucgnn")
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument("--stop-after-epoch", type=int)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--view")
    ev.add_argument("--root", default="data/ucgnn")
    ev.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ev.add_argument("--degrees", type=int, nargs="+", required=True)
    ev.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    ev.add_argument("--count", type=int, help="graphs per family and size (default: frozen M6B counts)")
    ev.add_argument("--split", default="final_test")
    ev.add_argument("--namespace", type=int, default=20261014)
    ev.add_argument("--max-rounds", type=int, default=256)
    ev.add_argument("--teacher-rounds", action="store_true")
    sm = sub.add_parser("smoke")
    sm.add_argument("--root", default="data/ucgnn-smoke")
    sm.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for command in ("stop",):
        p = sub.add_parser(command)
        p.add_argument("--root", default="data/ucgnn")
    args = parser.parse_args()
    if args.command == "plan":
        write_plans(args.output)
    elif args.command == "run":
        run_queue(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "stop":
        Path(args.root).mkdir(parents=True, exist_ok=True)
        (Path(args.root) / "STOP").touch()
        print("Stop requested; the current epoch or graph finishes and persists")


if __name__ == "__main__":
    main()
