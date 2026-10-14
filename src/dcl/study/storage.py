from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import torch


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    atomic_write(path, lambda f: f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()))


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, value):
    atomic_write(path, lambda f: torch.save(value, f))


def file_hash(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def provenance():
    from importlib.metadata import version
    import platform
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "source_hashes": {str(p): file_hash(p) for p in sorted(Path("src/dcl").rglob("*.py"))},
            "versions": {name: version(name) for name in ("torch", "numpy", "networkx", "scipy", "matplotlib")},
            "python": platform.python_version(), "hostname": platform.node(), "created_utc": now()}


@contextmanager
def training_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Global across different study roots: no accidental second GPU trainer.
    lock_path = Path.home() / ".cache/dcl-m6-gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another study training/evaluation process holds the GPU lock") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def resources(root, min_disk_gb=10, min_ram_gb=2, device="cuda"):
    available_ram = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_ram = int(line.split()[1]) * 1024
    snapshot = {"disk_free_gb": shutil.disk_usage(root).free / 2**30, "ram_available_gb": available_ram / 2**30,
                "cuda_available": torch.cuda.is_available()}
    import resource
    snapshot["process_peak_ram_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30
    if snapshot["disk_free_gb"] < min_disk_gb or snapshot["ram_available_gb"] < min_ram_gb:
        raise RuntimeError(f"Resource threshold violated: {snapshot}")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU fallback for the main queue")
        free, total = torch.cuda.mem_get_info()
        snapshot.update(gpu=torch.cuda.get_device_name(), gpu_free_gb=free / 2**30, gpu_total_gb=total / 2**30)
        snapshot.update(gpu_allocated_gb=torch.cuda.memory_allocated() / 2**30, gpu_reserved_gb=torch.cuda.memory_reserved() / 2**30)
        if free < 512 * 2**20:
            raise RuntimeError("Less than 512 MiB GPU memory available")
    return snapshot


class Registry:
    def __init__(self, root):
        Path(root).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(Path(root) / "registry.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, config TEXT NOT NULL, status TEXT NOT NULL, epoch INTEGER DEFAULT 0, updates INTEGER DEFAULT 0, updated TEXT, detail TEXT)")

    def add(self, config):
        value = json.dumps(config.to_dict(), sort_keys=True)
        row = self.db.execute("SELECT config FROM runs WHERE id=?", (config.id,)).fetchone()
        if row and row[0] != value:
            raise RuntimeError("Experiment identifier collision")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO runs (id,config,status,updated) VALUES (?,?,?,?)", (config.id, value, "pending", now()))

    def set(self, config, status, epoch=0, updates=0, detail=""):
        self.add(config)
        with self.db:
            self.db.execute("UPDATE runs SET status=?,epoch=?,updates=?,updated=?,detail=? WHERE id=?", (status, epoch, updates, now(), detail, config.id))

    def rows(self):
        self.db.row_factory = sqlite3.Row
        return [dict(row) for row in self.db.execute("SELECT * FROM runs ORDER BY rowid")]
