"""Finite counterfactual banks, component accounting and replayable witnesses."""
from collections import Counter
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np

from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import DETECT, LINIAL, MAX_FIELD, named_seed
from dcl.samplecomplexity.data import array_hash, components
from dcl.samplecomplexity.protocol import graph_for
from dcl.study.storage import atomic_json, file_hash
from .oracles import confirmation, proposal_bank, stage_targets
from .states import trajectory, replay


def training_graph(config, seed, t, coverage):
    families = ["cubic", "irregular", "bipartite", "path", "cycle", "symmetric"]
    if coverage:
        families += ["edgeless", "mixed", "cliques-and-isolates"]
    family = families[t % len(families)]
    n = config["training_sizes"][(t // len(families)) % len(config["training_sizes"])]
    gs = named_seed(config["namespace"], "training", seed, "graph", t)
    if family == "edgeless":
        g = nx.empty_graph(n)
    elif family in ("mixed", "cliques-and-isolates"):
        core = nx.path_graph(n - max(2, n // 4)) if family == "mixed" else nx.disjoint_union_all(
            [nx.complete_graph(4) for _ in range(max(1, (n - 2) // 4))])
        g = nx.disjoint_union(core, nx.empty_graph(n - len(core)))
        order = np.random.default_rng(gs).permutation(n)
        g = nx.relabel_nodes(g, dict(enumerate(map(int, order))))
    else:
        g = graph_for(family, n, gs)
    return g, family, gs


def acquire(config, teacher, seed, budget, directory, coverage=True, alternate=False):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    identity = dict(namespace=config["namespace"], teacher=teacher, seed=seed, native_budget=budget,
                    coverage=coverage, alternate_prefixes=alternate, variants=4, interface="multicf-bank-v1")
    target, manifest_path = directory / "dataset.npz", directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if identity != manifest["identity"] or file_hash(target) != manifest["sha256"]:
            raise RuntimeError("Existing acquisition differs")
        return manifest
    buffers = {k: [] for k in ["a0", "a1", "a2", "a3", "native", "p", "c", "cf_p", "cf_c", "cf_y", "trajectory", "node", "step"]}
    cost, cf_cost, t, simulator_steps = 0, 0, 0, 0
    counts = Counter(); traces = []; witnesses = []
    rng = np.random.default_rng(named_seed(config["namespace"], "counterfactual-bank", seed))
    tick, cpu = time.perf_counter(), time.process_time()
    while cost < budget - 2:
        g, family, gs = training_graph(config, seed, t, coverage)
        tape = named_seed(config["namespace"], "training", seed, "tape", t)
        int_seed = named_seed(config["namespace"], "alternative-prefix", seed, t) if alternate and t % 3 == 2 else None
        start = cost
        for p, s, exact, witness in trajectory(g, teacher, tape, int_seed):
            simulator_steps += 1
            phase_cost = len(components(s.phase))
            roots = np.flatnonzero(s.permission)[:(budget - cost) // phase_cost]
            if not len(roots):
                break
            arrays = encode_roots(s, p.neighbors, roots)
            native = np.column_stack((exact.color[roots], exact.auxiliary[roots], exact.mark[roots], np.full(len(roots), s.phase)))
            ps = np.zeros((len(roots), 4), bool); cs = np.zeros((len(roots), 4), np.int64)
            cf_p = np.zeros((len(roots), 4, 4), bool); cf_c = np.zeros((len(roots), 4, 4), np.int64)
            cf_y = np.zeros((len(roots), 4), np.float32)
            if s.phase not in (LINIAL, DETECT):
                targets = stage_targets(arrays, native, teacher)
                ps, cs = targets.participate, targets.candidate
                native_y = confirmation(arrays, teacher, ps, cs)
                if not np.array_equal(native_y > .5, native[:, 0] > 0):
                    raise RuntimeError("Native/continuation differential failed")
                cf_p, cf_c, cf_y = proposal_bank(arrays, native, teacher, rng)
                cf_cost += len(roots) * 4
            for j, a in enumerate(arrays):
                buffers[f"a{j}"].append(a)
            for k, a in [("native", native), ("p", ps), ("c", cs), ("cf_p", cf_p), ("cf_c", cf_c), ("cf_y", cf_y)]:
                buffers[k].append(a)
            buffers["trajectory"].append(np.full(len(roots), t))
            buffers["node"].append(roots)
            buffers["step"].append(np.full(len(roots), p.steps))
            for c in components(s.phase):
                counts[c] += len(roots)
            cost += phase_cost * len(roots)
            # At least one fully replayable example per trajectory/phase,
            # including actual interventions and the reason for every label.
            if len(witnesses) < 210 or s.phase in (LINIAL, DETECT):
                q, checked = replay(witness)
                if array_hash(*encode_roots(checked, q.neighbors, roots[:1])) != array_hash(*(a[:1] for a in arrays)):
                    raise RuntimeError("Witness encoding differs")
                witnesses.append(dict(**witness, root=int(roots[0]), native=native[0].tolist(),
                    p=ps[0].tolist(), c=cs[0].tolist(), cf_p=cf_p[0].tolist(), cf_c=cf_c[0].tolist(),
                    cf_y=cf_y[0].tolist(), family=family, graph_seed=gs))
            if cost >= budget - 2:
                break
        traces.append(dict(trajectory=t, family=family, graph_seed=gs, tape_seed=tape, n=len(g),
                           native_labels=cost-start, alternate=int_seed is not None))
        t += 1
    packed = {k: np.concatenate(v) for k, v in buffers.items()}
    np.savez_compressed(target, **packed)
    witness_path = directory / "witnesses.json"
    atomic_json(witness_path, witnesses)
    manifest = dict(identity=identity, native_labels=cost, counterfactual_labels=cf_cost,
        total_acquired_component_labels=cost+cf_cost, by_native_component=dict(counts),
        by_counterfactual_component={"confirmation": cf_cost}, examples=len(packed["native"]),
        trajectories=traces, teacher_simulator_steps=simulator_steps, sha256=file_hash(target),
        witness_sha256=file_hash(witness_path), stored_bytes=target.stat().st_size+witness_path.stat().st_size,
        uncompressed_bytes=sum(a.nbytes for a in packed.values()), generation_seconds=time.perf_counter()-tick,
        cpu_seconds=time.process_time()-cpu, gpu_seconds=0.,
        accounting="Native bundles acquired once; every one of four stored counterfactual confirmation queries is charged, including coincident queries (conservative). Simulator steps and encoding are separate. Shared bank counted once for acquisition, separately for each model's access.")
    atomic_json(manifest_path, manifest)
    return manifest


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_hash(directory / "dataset.npz") != manifest["sha256"]:
        raise RuntimeError("Data hash differs")
    with np.load(directory / "dataset.npz", allow_pickle=False) as raw:
        packed = {k: raw[k] for k in raw.files}
    return packed, manifest
