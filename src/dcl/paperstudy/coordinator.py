"""Persistent, resumable, resource-budgeted study queue. No concurrent trainers."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from dcl.study.storage import atomic_json, now
from .config import ROOT, PYTHON, PROTOCOL_COMMIT, SEEDS, Run, protocol_hash
from .models import PRIMARY


def status(manifest):
    states = {}
    for t in manifest["tasks"]:
        states[t["status"]] = states.get(t["status"], 0) + 1
    running = next((t for t in manifest["tasks"] if t["status"] == "running"), None)
    message = ["# UCGNN paper study", "", f"Updated UTC: {now()}", "",
               f"Protocol frozen at `{PROTOCOL_COMMIT}`; SHA256 `{manifest['protocol_sha256']}`.",
               f"Queue: **{manifest['status']}**; accumulated active GPU process hours: {manifest['active_seconds']/3600:.4f} / 36.",
               f"Task states: `{json.dumps(states, sort_keys=True)}`.", ""]
    if running:
        message.append(f"Running `{running['id']}` (pid {running.get('pid')}); log `{running['log']}`.")
        if running["action"] == "train":
            p = ROOT / "runs" / running["run"] / "progress.json"
            if p.exists():
                message.append(f"Progress: `{p.read_text().strip()}`")
    message += ["", "Historical directories are read-only. One GPU worker under global flock;",
                "RTX 4060 8 GiB; process allocator cap 4 GiB; no external services stopped.", "",
                "```bash", "tmux attach -t ucgnn-paper-study",
                f"{PYTHON} -m json.tool {ROOT}/manifest.json",
                "tail -n 15 data/ucgnn-paper-study/logs/*.txt",
                "bash experiments/ucgnn-paper-study resume", "```", "",
                "tmux survives disconnection. After a container restart run the resume command;",
                "there is no claim of automatic host/container reboot recovery.", "",
                "Training pilots and all promotion decisions precede confirmatory evaluation.",
                "Final results, mathematical analysis and manuscript status are in",
                "`results/ucgnn-paper-study/REPORT.md` when generated."]
    Path("STATUS.md").write_text("\n".join(message) + "\n")


def persist(manifest):
    manifest["updated_utc"] = now()
    atomic_json(ROOT / "manifest.json", manifest)
    status(manifest)


def train_task(treatment, seed, pilot=False):
    run = Run(treatment, seed)
    return {"id": f"train-{run.id}", "action": "train", "run": run.id, "treatment": treatment,
            "seed": seed, "pilot": pilot, "status": "pending", "attempts": [], "timeout": 2400,
            "arguments": ["train", "--treatment", treatment, "--seed", str(seed)]}


def initial_manifest():
    # Check the frozen document against the actual commit, not only current text.
    frozen = subprocess.check_output(["git", "show", f"{PROTOCOL_COMMIT}:docs/UCGNN_PAPER_PROTOCOL.md"])
    import hashlib
    if hashlib.sha256(frozen).hexdigest() != protocol_hash():
        raise RuntimeError("Protocol differs from preregistration commit")
    return {"protocol_commit": PROTOCOL_COMMIT, "protocol_sha256": protocol_hash(), "active_seconds": 0.,
            "created_utc": now(), "status": "running", "promotions_final": False, "evaluation_scheduled": False,
            "secondary_scheduled": False, "budget_hours": 36,
            "tasks": [train_task(v, SEEDS[0], True) for v in PRIMARY] + [train_task("own_only", SEEDS[0])]}


def advance(manifest):
    from .training import promotion
    pilots = [t for t in manifest["tasks"] if t.get("pilot")]
    if not manifest["promotions_final"] and all(t["status"] in ("complete", "technical_failure") for t in pilots):
        promotions = {}
        for task in pilots:
            p = ROOT / "runs" / task["run"] / "epoch-120.pt"
            decision = promotion(p) if p.exists() else {"promoted": False, "technical_failure": True}
            promotions[task["treatment"]] = decision
            if decision["promoted"]:
                manifest["tasks"].extend(train_task(task["treatment"], seed) for seed in SEEDS[1:])
        atomic_json(ROOT / "promotions.json", {"frozen_before_test": True, "decided_utc": now(), "decisions": promotions})
        manifest["promotions_final"] = True
    training = [t for t in manifest["tasks"] if t["action"] == "train"]
    if (manifest["promotions_final"] and not manifest["evaluation_scheduled"]
            and all(t["status"] in ("complete", "technical_failure") for t in training)):
        if not (ROOT / "EVALUATION_READY").exists():
            manifest["status"] = "waiting_evaluation_implementation"
            return
        from .evaluation import schedule_evaluations
        manifest["tasks"].extend(schedule_evaluations())
        manifest["evaluation_scheduled"] = True
    if manifest["evaluation_scheduled"] and all(t["status"] in ("complete", "technical_failure") for t in manifest["tasks"]):
        if not manifest["secondary_scheduled"]:
            manifest["tasks"].append(train_task("cyclic_teacher", 41005))
            manifest["secondary_scheduled"] = True
        elif not manifest.get("cyclic_evaluation_scheduled"):
            from .evaluation import schedule_evaluations
            manifest["tasks"].extend(schedule_evaluations(cyclic_only=True))
            manifest["cyclic_evaluation_scheduled"] = True
        elif not manifest.get("mechanisms_scheduled"):
            if not (ROOT / "MECHANISMS_READY").exists():
                manifest["status"] = "waiting_mechanisms_implementation"
                return
            manifest["tasks"].append({"id":"mechanisms", "action":"mechanisms", "status":"pending", "attempts":[],
                                      "timeout":1800, "arguments":["mechanisms"]})
            manifest["mechanisms_scheduled"] = True
        else:
            manifest["status"] = "complete"


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = ROOT / "manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else initial_manifest()
        if manifest["protocol_sha256"] != protocol_hash():
            raise RuntimeError("Protocol hash mismatch")
        for task in manifest["tasks"]:
            if task["status"] == "running":
                # A surviving child must not be duplicated on coordinator restart.
                if Path(f"/proc/{task.get('pid', -1)}").exists():
                    raise RuntimeError(f"Recorded worker {task['pid']} still exists; inspect before recovery")
                task["status"] = "pending"
                task["attempts"].append({"status":"interrupted_process", "utc":now()})
        manifest["status"] = "running"
        stop = [False]
        signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *a: stop.__setitem__(0, True))
        while not stop[0]:
            advance(manifest)
            if manifest["status"] == "complete":
                persist(manifest)
                print("QUEUE COMPLETE", flush=True)
                return
            remaining = 36 * 3600 - manifest["active_seconds"]
            if remaining <= 0:
                manifest["status"] = "budget_exhausted"
                persist(manifest)
                return
            task = next((t for t in manifest["tasks"] if t["status"] == "pending"), None)
            if task is None:
                persist(manifest)
                time.sleep(5)
                continue
            # Resource polling occurs before starting/accounting the GPU child.
            free = int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip().splitlines()[0])
            global_lock = Path.home() / ".cache/dcl-m6-gpu.lock"
            global_lock.parent.mkdir(parents=True,exist_ok=True)
            with global_lock.open("a+") as probe:
                try:
                    fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    fcntl.flock(probe,fcntl.LOCK_UN)
                    busy=False
                except BlockingIOError:
                    busy=True
            if free < 1024 or busy:
                manifest["status"] = "waiting_external_resources"
                persist(manifest)
                time.sleep(10)
                continue
            logfile = ROOT / "logs" / f"{task['id']}-attempt{len(task['attempts'])+1}.txt"
            task.update(status="running", log=str(logfile), started_utc=now())
            manifest["status"] = "running"
            env = {**os.environ, "PYTHONPATH":"src", "CUBLAS_WORKSPACE_CONFIG":":4096:8",
                   "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"}
            with logfile.open("w") as log:
                child = subprocess.Popen([PYTHON, "-u", "-m", "dcl.paperstudy.worker", *task["arguments"]],stdout=log,stderr=subprocess.STDOUT,env=env)
                task["pid"] = child.pid
                persist(manifest)
                started, accounted, timed_out = time.monotonic(), time.monotonic(), False
                while child.poll() is None:
                    time.sleep(2)
                    current = time.monotonic()
                    manifest["active_seconds"] += current-accounted
                    accounted = current
                    persist(manifest)
                    if current-started > min(task["timeout"], remaining) or stop[0]:
                        timed_out = not stop[0]
                        child.terminate()
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        break
                manifest["active_seconds"] += time.monotonic()-accounted
            task["attempts"].append({"returncode":child.returncode,"seconds":time.monotonic()-started,
                                     "timed_out":timed_out,"log":str(logfile),"ended_utc":now()})
            if stop[0]:
                task["status"] = "pending"
                manifest["status"] = "stopped_recoverable"
                persist(manifest)
                return
            if timed_out:
                task["status"] = "technical_failure"
            elif child.returncode != 0:
                task["status"] = "error"
                manifest["status"] = "halted_programming_or_resource_error"
                persist(manifest)
                raise RuntimeError(f"Worker failed: {task['id']}; inspect {logfile}")
            elif task["action"] == "train":
                expected = ROOT / "runs" / task["run"] / "epoch-120.pt"
                if not expected.exists():
                    raise RuntimeError(f"Worker exited without required checkpoint: {expected}")
                task["status"] = "complete"
            else:
                task["status"] = "complete"
            print(f"{now()} {task['id']} {task['status']} active_hours={manifest['active_seconds']/3600:.3f}",flush=True)
            persist(manifest)


if __name__ == "__main__":
    run()
