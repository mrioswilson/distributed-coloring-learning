"""Immutable native trajectories, shared by initialization replicas."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from dcl.study.config import digest
from dcl.study.graphs import generate_graph, fingerprint
from dcl.study.storage import atomic_json, file_hash
from .model import FEATURES, RELATIONS, encode_snapshot
from .programs import ColoringProgram, DETECT, LINIAL, teacher_choices, named_seed


@dataclass
class Pool:
    x: np.ndarray
    relations: np.ndarray
    targets: np.ndarray
    graph_index: np.ndarray
    neighbors: np.ndarray
    roots: np.ndarray
    manifest: dict

    def batch(self, root_positions):
        flat = self.roots[root_positions].astype(np.int64)
        width = self.x.shape[1]
        state, root = flat // width, flat % width
        neighbors = self.neighbors[self.graph_index[state]]
        centers = np.concatenate((root[:, None], neighbors[np.arange(len(root)), root]), axis=1)
        adjacent = np.take_along_axis(neighbors, centers[..., None], axis=1)
        center_x = self.x[state[:, None], centers].copy()
        adjacent_x = self.x[state[:, None, None], adjacent].copy()
        relations = self.relations[state[:, None], centers].astype(np.float32)
        labels = self.targets[state, root].astype(np.int64)
        # Phase is stored separately in the target's fourth entry.
        return (center_x, adjacent_x, relations, adjacent < width - 1), labels


def build_pool(config, root, teacher, split):
    if split not in ("train", "validation"):
        raise ValueError("Final test must never supply training examples")
    folder = Path(root) / "pools" / teacher.lower() / split
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    expected = {"protocol": config["protocol"], "namespace": config["namespace"], "teacher": teacher,
                "split": split, "n": config["n"], "graphs": config["graphs"] if split == "train" else config["val_graphs"],
                "tapes": config["tapes"] if split == "train" else 1}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != expected:
            raise RuntimeError("Existing pool configuration differs")
        for name, expected_hash in manifest["hashes"].items():
            if file_hash(folder / manifest["attempt"] / name) != expected_hash:
                raise RuntimeError("Persistent native examples changed")
        return load_pool(folder, manifest)
    attempt = 1
    while (folder / f"attempt-{attempt}").exists():
        attempt += 1
    work = folder / f"attempt-{attempt}"
    work.mkdir()
    n = config["n"]
    states, roots, graphs, graph_indices, neighbor_tables, records = 0, [], [], [], [], []
    paths = {name: (work / name).open("xb") for name in ("x.bin", "relations.bin", "targets.bin")}
    try:
        for i in range(expected["graphs"]):
            graph_seed = named_seed(config["protocol"], config["namespace"], split, "graph", i)
            graph = generate_graph(n, 3, "REGULAR", graph_seed)
            record = {"graph_i": i, "graph_seed": graph_seed, "fingerprint": fingerprint(graph), "trajectories": []}
            for j in range(expected["tapes"]):
                tape_seed = named_seed(config["protocol"], config["namespace"], split, "tape", i, j)
                program = ColoringProgram(graph, teacher, tape_seed)
                if j == 0:
                    neighbor_tables.append(program.neighbors)
                start = states
                while not program.finished:
                    state = program.snapshot()
                    choices = teacher_choices(state, program.neighbors, teacher)
                    x, relations = encode_snapshot(state, program.neighbors)
                    labels = np.zeros((n + 1, 4), dtype=np.int16)
                    labels[:n] = np.column_stack((choices.color, choices.auxiliary, choices.mark,
                                                  np.full(n, state.phase)))
                    paths["x.bin"].write(x.tobytes())
                    paths["relations.bin"].write(relations.tobytes())
                    paths["targets.bin"].write(labels.tobytes())
                    roots.extend((states * (n + 1) + np.flatnonzero(state.permission)).tolist())
                    graph_indices.append(i)
                    states += 1
                    program.apply(state, choices)
                if program.failure is not None:
                    raise RuntimeError(f"Exact teacher failed: {teacher}/{i}/{j}: {program.failure}")
                record["trajectories"].append({"tape_seed": tape_seed, "steps": states - start})
            records.append(record)
            if (i + 1) % 200 == 0:
                print(f"pool {teacher} {split}: graphs={i+1} native_steps={states}", flush=True)
    finally:
        for handle in paths.values():
            handle.close()
    np.save(work / "roots.npy", np.asarray(roots, np.int32))
    np.save(work / "graph_index.npy", np.asarray(graph_indices, np.int32))
    np.save(work / "neighbors.npy", np.asarray(neighbor_tables, np.int32))
    manifest = {"config": expected, "attempt": work.name, "states": states, "decisions": len(roots),
                "records": records, "features": FEATURES, "relations": RELATIONS,
                "hashes": {p.name: file_hash(p) for p in work.iterdir()}}
    atomic_json(manifest_path, manifest)
    return load_pool(folder, manifest)


def load_pool(folder, manifest):
    work, n, states = folder / manifest["attempt"], manifest["config"]["n"], manifest["states"]
    return Pool(np.memmap(work / "x.bin", mode="r", dtype=np.float32, shape=(states, n + 1, FEATURES)),
                np.memmap(work / "relations.bin", mode="r", dtype=np.uint8, shape=(states, n + 1, 3, RELATIONS)),
                np.memmap(work / "targets.bin", mode="r", dtype=np.int16, shape=(states, n + 1, 4)),
                np.load(work / "graph_index.npy", mmap_mode="r"), np.load(work / "neighbors.npy", mmap_mode="r"),
                np.load(work / "roots.npy", mmap_mode="r"), manifest)
