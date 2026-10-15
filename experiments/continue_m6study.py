"""Run the remaining preregistered training stages sequentially.

Final testing starts after a separate validation review and protocol freeze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile


STAGES = (
    "phase-b-all-degrees",
    "references",
    "references-original-budget",
    "phase-c-epochs",
    "phase-c-updates",
    "phase-d",
)


def write_state(path, state):
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def plan_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pipeline(repo, root, stages=STAGES, resume=False, dry_run=False):
    plans = []
    for name in stages:
        path = repo / "configs/m6study" / f"{name}.json"
        value = json.loads(path.read_text())
        if value["protocol"] != "m6study-v1":
            raise ValueError("Unknown protocol")
        plans.append({"name": name, "path": str(path), "sha256": plan_hash(path),
                      "experiments": len(value["experiments"])})
    if dry_run:
        print(json.dumps({"root": str(root), "stages": plans, "includes_final_test": False}, indent=2))
        return 0

    root.mkdir(parents=True, exist_ok=True)
    stop_file, state_path = root / "STOP", root / "continuation-state.json"
    with (root / "continuation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another continuation is already running") from None
        if resume:
            stop_file.unlink(missing_ok=True)
        if stop_file.exists():
            print("Continuation is stopped; use --resume after the previous session ends", flush=True)
            return 0
        state = {"stages": plans, "completed_stages": [], "status": "starting", "stage": None,
                 "pid": os.getpid(), "final_test_started_by_pipeline": False}
        child = None
        requested = False

        def stop(signum, frame):
            nonlocal requested
            requested = True
            stop_file.touch()
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)

        original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in original_handlers:
            signal.signal(sig, stop)
        try:
            for plan in plans:
                if requested or stop_file.exists():
                    state["status"] = "paused"
                    write_state(state_path, state)
                    return 0
                if plan_hash(Path(plan["path"])) != plan["sha256"]:
                    raise RuntimeError("A preregistered plan changed during the continuation")
                state.update(stage=plan["name"], status="running")
                write_state(state_path, state)
                print(f"Starting {plan['name']}; completed experiments are skipped", flush=True)
                child = subprocess.Popen([str(repo / "experiments/m6study"), "run", "--plan", plan["path"],
                                          "--root", str(root)], cwd=repo)
                return_code = child.wait()
                child = None
                if return_code:
                    state.update(status="failed", return_code=return_code)
                    write_state(state_path, state)
                    return return_code if return_code > 0 else 1
                if requested or stop_file.exists():
                    state["status"] = "paused"
                    write_state(state_path, state)
                    return 0
                state["completed_stages"].append(plan["name"])
                write_state(state_path, state)
            state.update(status="awaiting_validation_review", stage=None)
            write_state(state_path, state)
            print("Training stages complete. Review validation and decide budget extensions before freezing the final test.", flush=True)
            return 0
        except Exception as error:
            state.update(status="failed", error=str(error))
            write_state(state_path, state)
            raise
        finally:
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/m6study-release")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    if not root.is_absolute():
        root = repo / root
    return run_pipeline(repo, root.resolve(), resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
