"""Lossless packing of the existing rooted training examples into resident RAM."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from dcl.study.config import digest
from dcl.study.storage import atomic_json, file_hash


@dataclass
class ResidentRoots:
    arrays: tuple
    targets: np.ndarray
    roots: np.ndarray
    manifest: dict
    cache_manifest: dict

    def batch(self, positions):
        arrays = tuple(a[positions].astype(np.float32) if i == 2 else a[positions]
                       for i, a in enumerate(self.arrays))
        return arrays, self.targets[positions].astype(np.int64)


def resident_roots(base, directory, chunk=4096):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    source_hash = digest(base.manifest)
    if not manifest_path.exists():
        attempt = 1
        while (directory / f"attempt-{attempt}").exists():
            attempt += 1
        work = directory / f"attempt-{attempt}"
        work.mkdir()
        sample, labels = base.batch(np.arange(1))
        n = len(base.roots)
        dtypes = [np.float32, np.float32, np.uint8, np.bool_, np.int16]
        examples = list(sample) + [labels]
        maps = [np.lib.format.open_memmap(work / f"array-{i}.npy", mode="w+", dtype=dtype,
                         shape=(n,) + example.shape[1:])
                for i, (dtype, example) in enumerate(zip(dtypes, examples))]
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            values, targets = base.batch(np.arange(start, end))
            for value, target in zip(list(values) + [targets], maps):
                converted = value.astype(target.dtype)
                if not np.array_equal(value, converted):
                    raise RuntimeError("Cache conversion would change a training value")
                target[start:end] = converted
        for mapped in maps:
            mapped.flush(); mapped._mmap.close()
        metadata = {"kind": "lossless-root-context-v1", "source_manifest_digest": source_hash,
                    "attempt": work.name, "roots": n,
                    "hashes": {p.name: file_hash(p) for p in work.iterdir()}}
        atomic_json(manifest_path, metadata)
    metadata = json.loads(manifest_path.read_text())
    if metadata["source_manifest_digest"] != source_hash or metadata["roots"] != len(base.roots):
        raise RuntimeError("Resident cache belongs to another source pool")
    work = directory / metadata["attempt"]
    for name, expected in metadata["hashes"].items():
        if file_hash(work / name) != expected:
            raise RuntimeError("Resident training cache changed")
    arrays = [np.load(work / f"array-{i}.npy", allow_pickle=False) for i in range(5)]
    for array in arrays:
        array.setflags(write=False)
    return ResidentRoots(tuple(arrays[:4]), arrays[4], np.array(base.roots, copy=True), base.manifest, metadata)


def verify_resident_roots(base, cached, chunk=4096):
    """Check every input, mask and label against the authoritative source pool."""
    for start in range(0, len(base.roots), chunk):
        positions = np.arange(start, min(start + chunk, len(base.roots)))
        a, y = base.batch(positions)
        b, z = cached.batch(positions)
        for left, right in zip(a + (y,), b + (z,)):
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                raise RuntimeError("Resident cache differs from original examples")
    return len(base.roots)
