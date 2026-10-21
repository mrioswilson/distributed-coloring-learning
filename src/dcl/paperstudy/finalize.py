"""CPU-only report/PDF handoff, also when the GPU budget stops the queue."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

from dcl.study.storage import atomic_json, file_hash, now
from .config import ROOT


TERMINAL = {"complete", "budget_exhausted", "stopped_recoverable",
            "halted_programming_or_resource_error"}


def publish():
    from .report import build
    build()
    compiler = Path(".tools/tectonic")
    result = {"created_utc": now(), "manifest_sha256": file_hash(ROOT / "manifest.json"),
              "queue_status": json.loads((ROOT / "manifest.json").read_text())["status"]}
    if compiler.exists():
        with (ROOT / "logs/finalize-latex.txt").open("w") as output:
            compiled = subprocess.run([str(compiler.resolve()), "-X", "compile", "--keep-logs",
                                       "paper/ucgnn/main.tex"], stdout=output, stderr=subprocess.STDOUT,
                                      timeout=300, check=False)
        result["latex_exit_code"] = compiled.returncode
        if compiled.returncode:
            atomic_json(ROOT / "finalization.json", result)
            raise RuntimeError("LaTeX failed; inspect logs/finalize-latex.txt")
        result["pdf_sha256"] = file_hash("paper/ucgnn/main.pdf")
    else:
        result["latex_status"] = "compiler_unavailable"
    atomic_json(ROOT / "finalization.json", result)
    print(json.dumps(result), flush=True)


def watch():
    # Separate lock from the GPU worker and coordinator: reports consume no GPU.
    with (ROOT / "finalizer.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        absent_since = None
        while True:
            try:
                manifest = json.loads((ROOT / "manifest.json").read_text())
                with (ROOT / "coordinator.lock").open("a+") as probe:
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        alive = False
                    except BlockingIOError:
                        alive = True
                absent_since = None if alive else (absent_since or time.monotonic())
                orphaned = absent_since is not None and time.monotonic() - absent_since > 120
                if manifest["status"] in TERMINAL or orphaned:
                    time.sleep(5)  # Let the coordinator finish its final atomic writes.
                    publish()
                    return
            except Exception:
                traceback.print_exc()
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Publish current partial results immediately")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    publish() if args.once else watch()
