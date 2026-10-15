"""Execute validation-selected extensions, freeze, final tests, and reports."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess

from dcl.study.config import Config
from dcl.study.storage import atomic_json, file_hash, now
from review_m6study import read_plan


COUNTS = {"40": 100, "160": 100, "640": 100, "2560": 50, "10240": 25, "40960": 10}


def freeze_cohort(root, cohort_plan, review_path):
    import torch
    destination = root / "frozen-final-test.json"
    identity = {"validation_review_sha256": file_hash(review_path), "cohort_plan_sha256": file_hash(cohort_plan)}
    if destination.exists():
        saved = json.loads(destination.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise RuntimeError("The existing frozen test belongs to another review or cohort")
        return destination
    if any((root / "evaluations").glob("*/*/*.json")):
        raise RuntimeError("Refusing to create a new protocol after final testing started")
    models = []
    for config in read_plan(cohort_plan):
        directory = root / "runs" / config.id
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        if not last["complete"] or last["config"] != config.to_dict() or last["epoch"] != config.epochs:
            raise RuntimeError(f"Incomplete or mismatched training: {config.id}")
        choices = [("best", "best.pt"), ("epoch-120", "epoch-120.pt")]
        if config == Config(seed=config.seed):
            choices += [(f"epoch-{epoch:03d}", f"epoch-{epoch:03d}.pt") for epoch in (20, 40, 60, 80)]
        for policy, filename in choices:
            path = directory / filename
            selected = torch.load(path, map_location="cpu", weights_only=False)
            if selected["config"] != config.to_dict():
                raise RuntimeError("Selected checkpoint configuration differs")
            models.append({"id": config.id if policy == "best" else f"{config.id}-{policy}",
                           "config": config.to_dict(), "checkpoint": str(path.resolve()),
                           "checkpoint_policy": policy, "checkpoint_sha256": file_hash(path),
                           "selected_epoch": selected["epoch"],
                           "selection_score": selected["best_score"] if policy == "best" else None})
    if len({entry["id"] for entry in models}) != len(models):
        raise RuntimeError("Duplicate frozen checkpoint views")
    atomic_json(destination, {"protocol": "m6study-v1", "created_utc": now(), "models": models,
                "counts": COUNTS, "families": ["REGULAR", "IRREGULAR"], "max_rounds": 256,
                "test_seed_namespace": 20261014, "selection_uses_final_test": False, **identity})
    print(f"Frozen {len(models)} checkpoint views; {len(models) * 2 * sum(COUNTS.values())} graph evaluations", flush=True)
    return destination


def run_pipeline(repo, root, review_path, resume=False, dry_run=False):
    review = json.loads(review_path.read_text())
    if not review["review_complete"] or review["final_test_observations_used"] != 0:
        raise RuntimeError("A completed validation-only review is required")
    plans = review["plans"]
    for plan in plans.values():
        if file_hash(Path(plan["path"])) != plan["sha256"]:
            raise RuntimeError("A reviewed plan changed")
    extension = Path(plans["phase-c-extension"]["path"])
    cohort = Path(plans["phase-e-cohort"]["path"])
    frozen = root / "frozen-final-test.json"
    python, command = str(repo / ".venv/bin/python"), str(repo / "experiments/m6study")
    stages = [("extensions", [command, "run", "--plan", str(extension), "--root", str(root)]),
              ("verification", [python, str(repo / "experiments/review_m6study.py"), "verify", "--root", str(root),
                                "--plan", str(cohort), "--review", str(review_path)]),
              ("freeze", None),
              ("final_test", [command, "evaluate", "--root", str(root), "--freeze", str(frozen)]),
              ("reports", [python, str(repo / "experiments/report_m6study_cohorts.py"), "--root", str(root),
                           "--freeze", str(frozen)])]
    identity = {"review_sha256": file_hash(review_path), "plan_hashes": {key: value["sha256"] for key, value in plans.items()}}
    if dry_run:
        print(json.dumps({"root": str(root), "additional_trainings": review["additional_unique_training"],
                          "stages": [name for name, _ in stages], "includes_final_test": True, **identity}, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    stop_file, state_path = root / "STOP", root / "finalization-state.json"
    with (root / "continuation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another continuation is already running") from None
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        if previous and any(previous.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Finalization inputs changed since the previous attempt")
        if previous.get("status") == "complete":
            print("Finalization is already complete", flush=True)
            return 0
        if resume:
            stop_file.unlink(missing_ok=True)
        state = {**previous, **identity, "pid": os.getpid(), "status": "starting", "stage": None}
        state.pop("return_code", None)
        state.pop("error", None)
        child, requested = None, False

        def save():
            state["updated_utc"] = now()
            atomic_json(state_path, state)

        def stopped():
            if requested or stop_file.exists():
                state["status"] = "paused"
                save()
                return True
            return False

        def stop(signum, frame):
            nonlocal requested
            requested = True
            stop_file.touch()
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)

        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in handlers:
            signal.signal(sig, stop)
        try:
            if stopped():
                return 0
            for name, arguments in stages:
                if stopped():
                    return 0
                for plan in plans.values():
                    if file_hash(Path(plan["path"])) != plan["sha256"]:
                        raise RuntimeError("A reviewed plan changed during finalization")
                state.update(status="running", stage=name)
                save()
                print(f"Starting {name} at {now()}", flush=True)
                if name == "freeze":
                    freeze_cohort(root, cohort, review_path)
                    fingerprint = file_hash(frozen)
                    if state.get("freeze_sha256", fingerprint) != fingerprint:
                        raise RuntimeError("The frozen protocol changed")
                    state["freeze_sha256"] = fingerprint
                    save()
                else:
                    if name in ("final_test", "reports") and file_hash(frozen) != state["freeze_sha256"]:
                        raise RuntimeError("The frozen protocol changed")
                    child = subprocess.Popen(arguments, cwd=repo)
                    code = child.wait()
                    child = None
                    if stopped():
                        return 0
                    if code:
                        state.update(status="failed", return_code=code)
                        save()
                        return code if code > 0 else 1
            state.update(status="complete", stage=None)
            save()
            print("Final testing and scientific reports are complete", flush=True)
            return 0
        except Exception as error:
            state.update(status="failed", error=str(error))
            save()
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/m6study-release"))
    parser.add_argument("--review", type=Path, default=Path("data/m6study-release/reports/validation-review.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    return run_pipeline(repo, args.root.resolve(), args.review.resolve(), args.resume, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
