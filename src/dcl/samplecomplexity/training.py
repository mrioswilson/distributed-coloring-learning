"""Pure replay of stored component labels; new initialization for every B."""
from collections import Counter
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from dcl.multiteacher.programs import DETECT, LINIAL, named_seed
from dcl.multiteacher.experiment import weight_hash
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.study.storage import atomic_json, file_hash, save_checkpoint
from .data import load, components
from .model import QueryPolicy
from .protocol import cases, summarize_runs
from .evaluation import evaluate, teacher_state_fidelity


def initialize(teacher, architecture, seed):
    init = named_seed("sample-complexity-initialization-v1", seed) % 2**63
    random.seed(init); np.random.seed(init % 2**32); torch.manual_seed(init)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    return QueryPolicy(teacher, architecture)


def passive_loss(model, arrays, labels):
    """No labels recomputed from inputs or the learner's current proposals."""
    phase = int(labels[0, 0]); out = model(*tensor_arrays(arrays), phase=phase)
    if phase == LINIAL:
        return F.cross_entropy(out.scores, torch.from_numpy(labels[:, 5]))
    if phase == DETECT:
        return F.binary_cross_entropy_with_logits(out.commit, torch.from_numpy(labels[:, 4].astype(np.float32)))
    p = F.binary_cross_entropy_with_logits(out.state.p_logits[:, 0], torch.from_numpy(labels[:, 1].astype(np.float32)))
    c = F.cross_entropy(out.scores, torch.from_numpy(labels[:, 2]))
    confirm = F.binary_cross_entropy_with_logits(out.commit, torch.from_numpy(labels[:, 3].astype(np.float32)))
    return p + c + confirm


def train(config, teacher, architecture, seed, budget, dataset, directory, updates=None, dev=True, warmstart=None, prepared=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    updates = updates or config["updates"]
    identity = dict(teacher=teacher, architecture=architecture, seed=seed, budget=budget,
                    updates=updates, lr=config["lr"], batch=config["batch_size"])
    if warmstart is not None: identity["warmstart_sha256"]=file_hash(warmstart)
    record_path = directory / "record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["identity"] != identity or file_hash(directory / "final.pt") != record["checkpoint_sha256"]:
            raise RuntimeError("Existing training identity/checkpoint differs")
        return record
    arrays, labels, accounting = prepared if prepared is not None else load(dataset, budget)
    model = initialize(teacher, architecture, seed)
    if warmstart is not None: model.load_state_dict(torch.load(warmstart,weights_only=False)["model"])
    initial_hash = weight_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    order = np.random.default_rng(named_seed(config["namespace"], "batch-order", seed))
    phase_indices = {int(p): np.flatnonzero(labels[:, 0] == p) for p in np.unique(labels[:, 0])}
    # Fixed, phase-balanced sampling; no final/development statistics choose it.
    phases = sorted(phase_indices)
    visits = np.zeros(len(labels), np.int64); reuse = Counter(); history = []
    tick = time.perf_counter(); cpu_tick = time.process_time()
    for update in range(updates):
        phase = phases[update % len(phases)]
        positions = order.choice(phase_indices[phase], config["batch_size"], replace=True)
        np.add.at(visits, positions, 1)
        for comp in components(phase): reuse[comp] += len(positions)
        optimizer.zero_grad(set_to_none=True)
        loss = passive_loss(model, tuple(a[positions] for a in arrays), labels[positions])
        if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite loss")
        loss.backward(); optimizer.step()
        if (update + 1) % 100 == 0 or update + 1 == updates:
            history.append(dict(update=update+1, loss=float(loss.detach())))
    seconds, cpu_seconds = time.perf_counter()-tick, time.process_time()-cpu_tick
    record = dict(identity=identity, accounting=accounting, initial_hash=initial_hash, weights_hash=weight_hash(model),
                  parameters=sum(p.numel() for p in model.parameters()), training_seconds=seconds,
                  cpu_seconds=cpu_seconds, gpu_seconds=0., device="cpu", history=history,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  parameter_updates=updates, epochs=None, effective_passes=updates*config["batch_size"]/len(labels),
                  component_presentations=dict(reuse), acquired_examples_ever_used=int((visits>0).sum()),
                  repeated_example_presentations=int(np.maximum(visits-1, 0).sum()),
                  per_example_visit_hash=__import__("hashlib").sha256(visits.tobytes()).hexdigest(),
                  optimization="Adam; cyclic phase strata, root bundles sampled with replacement; no full epochs",
                  flops_estimate=None, flops_note="Not measured; time and updates retained separately, no FLOP comparability claim")
    save_checkpoint(directory / "final.pt", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), numpy_rng=order.bit_generator.state,
                    python_rng=random.getstate(), record=record, config=config))
    record["checkpoint_sha256"] = file_hash(directory / "final.pt")
    if dev:
        suite = cases(config, "development", seed)
        rows, elapsed = evaluate(model, teacher, suite)
        atomic_json(directory / "development.json", rows)
        record["development"] = summarize_runs(rows, [m["case_id"] for _, m in suite])
        record["development_seconds"] = elapsed
        record["fidelity"] = teacher_state_fidelity(model, teacher, suite, config["fidelity_decisions"])
    atomic_json(record_path, record)
    print(json.dumps(dict(complete=identity, seconds=round(seconds, 2),
                          development=record.get("development", {}).get("success_rate"))), flush=True)
    return record
