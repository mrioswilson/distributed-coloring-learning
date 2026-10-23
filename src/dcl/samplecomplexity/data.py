"""Acquire immutable passive labels once. Optimizers cannot call the teacher."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, DETECT, LINIAL, MAX_FIELD, named_seed, teacher_choices
from dcl.multiteacher_e1.supervision import labels_from_inputs
from dcl.study.graphs import fingerprint
from dcl.study.storage import atomic_json, file_hash
from .protocol import graph_for, horizon, digest


def components(phase):
    return ("linial_point",) if phase == LINIAL else (("conflict_mark",) if phase == DETECT else
            ("participation", "proposal", "confirmation"))


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def acquire(config, teacher, seed, budget, directory, split="training", policy=None):
    if split not in ("training", "pilot"):
        raise ValueError("Evaluation data must not enter the teacher dataset")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    target = directory / "dataset.npz"
    identity = dict(namespace=config["namespace"], teacher=teacher, seed=seed, budget=budget,
                    split=split, sizes=config["training_sizes"], interface="components-passive-v1")
    if policy is not None:
        from dcl.multiteacher.experiment import weight_hash
        identity.update(interface="components-interactive-v1", policy_sha256=weight_hash(policy))
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity or file_hash(target) != manifest["sha256"]:
            raise RuntimeError("Cached teacher data differ")
        return manifest
    batches = [[], [], [], []]; labels, trajectories, nodes, steps, state_ids = [], [], [], [], []
    global_hashes, local_hashes, graphs, records = set(), set(), {}, []
    queries, counts, simulations = 0, Counter(), 0
    t = 0
    while queries < budget:
        # Small/intermediate diverse families. Evaluation sizes are disjoint.
        family = ("cubic", "irregular", "bipartite", "path", "cycle", "symmetric")[t % 6]
        n = config["training_sizes"][(t // 6) % len(config["training_sizes"])]
        gs = named_seed(config["namespace"], split, seed, "graph", t)
        graph = graph_for(family, n, gs)
        tape = named_seed(config["namespace"], split, seed, "tape", t)
        p = ColoringProgram(graph, teacher, tape, max_rounds=horizon(teacher, n))
        fp = fingerprint(graph); graphs[fp] = dict(n=n, family=family, graph_seed=gs, edges=list(graph.edges()))
        start = queries
        while not p.finished and queries < budget:
            state = p.snapshot()
            if policy is not None:
                from .interactive import admissible
                if not admissible(p, state):
                    p.failure="inadmissible_interactive_state"; p.finished=True; break
            choices = teacher_choices(state, p.neighbors, teacher); simulations += 1
            cost = len(components(state.phase))
            roots = np.flatnonzero(state.permission)
            capacity = (budget - queries) // cost
            if capacity == 0:
                break
            selected = roots[:capacity]
            arrays = encode_roots(state, p.neighbors, selected)
            native = np.column_stack((choices.color[selected], choices.auxiliary[selected], choices.mark[selected],
                                      np.full(len(selected), state.phase)))
            lab = np.zeros((len(selected), 6), np.int64); lab[:, 0] = state.phase
            if state.phase == LINIAL:
                lab[:, 5] = native[:, 1] // MAX_FIELD
            elif state.phase == DETECT:
                lab[:, 4] = native[:, 2]
            else:
                target_components = labels_from_inputs(arrays, native, teacher)
                lab[:, 1] = target_components.participate[:, 0]
                lab[:, 2] = target_components.candidate[:, 0]
                lab[:, 3] = native[:, 0] > 0
            for buf, a in zip(batches, arrays): buf.append(a)
            labels.append(lab); trajectories.extend([t] * len(selected)); nodes.extend(selected.tolist())
            steps.extend([p.steps] * len(selected))
            gh = array_hash(state.colors, state.aux, state.marks, state.ids, state.slots,
                            state.permission, state.draws, state.preferences, np.array([state.phase, state.bound, state.target]))
            global_hashes.add(gh)
            state_ids.extend([gh] * len(selected))
            for j in range(len(selected)):
                local_hashes.add(array_hash(*(a[j] for a in arrays)))
            for c in components(state.phase): counts[c] += len(selected)
            queries += len(selected) * cost
            if policy is None:
                p.apply(state, choices)
            else:
                import torch
                from dcl.multiteacher_e1.experiment import predictions
                from dcl.multiteacher.programs import Choices
                from .evaluation import invariant_after
                full_roots=np.flatnonzero(state.permission)
                with torch.inference_mode():
                    pred=predictions(policy,encode_roots(state,p.neighbors,full_roots),np.full(len(full_roots),state.phase))
                proposed=Choices(np.zeros(n,int),np.zeros(n,int),np.zeros(n,bool))
                for dest,values in zip((proposed.color,proposed.auxiliary,proposed.mark),pred): dest[full_roots]=values
                before,old_aux=p.colors.copy(),p.aux.copy()
                p.apply(state,proposed)
                if not invariant_after(p,state,before,old_aux):
                    p.failure="invariant";p.finished=True
            if len(selected) < len(roots): break
        records.append(dict(trajectory=t, graph=fp, tape_seed=tape, queries=queries-start,
                            completed=p.finished, native_steps=p.steps,
                            failure=p.failure,
                            rng_state=p.rng.bit_generator.state,
                            preference_rng_state=p.preference_rng.bit_generator.state))
        if p.failure is not None and policy is None:
            raise RuntimeError(f"Exact training teacher failed; cannot discard trajectory: {teacher}/{t}/{p.failure}")
        t += 1
        if budget - queries < 3:
            break
    arrays = {f"a{i}": np.concatenate(buf) for i, buf in enumerate(batches)}
    arrays.update(labels=np.concatenate(labels), trajectory=np.asarray(trajectories),
                  node=np.asarray(nodes), native_step=np.asarray(steps), global_state=np.asarray(state_ids, dtype="S64"))
    np.savez_compressed(target, **arrays)
    manifest = dict(identity=identity, queries=queries, by_component=dict(counts), examples=len(trajectories),
                    independent_trajectories=t, training_graphs=len(graphs), distinct_local_states=len(local_hashes),
                    distinct_global_states=len(global_hashes), unique_stored_bytes=target.stat().st_size,
                    uncompressed_bytes=sum(a.nbytes for a in arrays.values()), sha256=file_hash(target),
                    graphs=graphs, trajectories=records, teacher_simulator_steps=simulations,
                    accounting="Each stored root component is acquired once; exact trajectory simulation is separately recorded. No counterfactual or new gradient-time labels.")
    atomic_json(manifest_path, manifest)
    return manifest


@lru_cache(maxsize=1)
def _read_arrays(path, expected_hash):
    if file_hash(path) != expected_hash: raise RuntimeError("Teacher data hash mismatch")
    with np.load(path, allow_pickle=False) as raw:
        return {k: raw[k] for k in raw.files}


def load(directory, budget):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    raw = _read_arrays(str(directory / "dataset.npz"), manifest["sha256"])
    labels = raw["labels"]
    costs = np.array([len(components(int(phase))) for phase in labels[:, 0]])
    take = int(np.searchsorted(np.cumsum(costs), budget, side="right"))
    if take == 0:
        raise ValueError("Budget cannot acquire one complete component bundle")
    counts = Counter()
    for phase in labels[:take, 0]: counts.update(components(int(phase)))
    arrays = tuple(raw[f"a{i}"][:take] for i in range(4))
    accounting = dict(budget=budget, queries=int(costs[:take].sum()), by_component=dict(counts),
                      examples=take, independent_trajectories=int(len(np.unique(raw["trajectory"][:take]))),
                      distinct_local_states=len({array_hash(*(a[i] for a in arrays)) for i in range(take)}),
                      data_sha256=manifest["sha256"], prefix_sha256=array_hash(*arrays, labels[:take]),
                      last_trajectory=int(raw["trajectory"][take-1]),
                      dataset_manifest_sha256=file_hash(directory / "manifest.json"))
    acquired_trajectories=set(map(int, raw["trajectory"][:take]))
    accounting.update(training_graphs=len({r["graph"] for r in manifest["trajectories"] if r["trajectory"] in acquired_trajectories}),
                      distinct_global_states=len(np.unique(raw["global_state"][:take])),
                      unique_stored_bytes=sum(a.nbytes for a in arrays)+labels[:take].nbytes,
                      dataset_cache_bytes=manifest["unique_stored_bytes"])
    return arrays, labels[:take], accounting
