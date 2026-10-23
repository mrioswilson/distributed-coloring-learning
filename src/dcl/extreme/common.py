from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def weights_hash(model):
    """Includes names, shapes, dtypes and exact bytes of parameters AND buffers."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def provenance():
    import torch
    import numpy
    import networkx
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    sources = {str(p.relative_to(root)): file_hash(p) for p in sorted((root / "src/dcl").rglob("*.py"))}
    return {"utc": now(), "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "source_hash": digest(sources), "source_hashes": sources, "torch": torch.__version__,
            "numpy": numpy.__version__, "networkx": networkx.__version__, "cuda": torch.version.cuda}
