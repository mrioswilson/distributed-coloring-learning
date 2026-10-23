"""Balanced native teacher x palette architecture x training-seed experiment.

CPU only. Cost pilots select an equal update budget before the scientific pools
are generated. Final checkpoints and all failure outcomes are retained.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import networkx as nx
import numpy as np
import torch

from dcl.integration.models import ARCHITECTURES, RootedPalettePolicy
from dcl.multiteacher.data import build_pool
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, TEACHERS, named_seed, teacher_choices
from dcl.multiteacher.experiment import weight_hash
from dcl.multiteacher_e1.experiment import rollout_many, validate_local
from dcl.multiteacher_e1.supervision import loss_one, phase_groups
from dcl.study.graphs import generate_graph, fingerprint
from dcl.study.storage import atomic_json, file_hash, save_checkpoint
from dcl.behavior_statistics import write_csv

REPO = Path(__file__).resolve().parents[3]
OUT = Path("results/integration/multiteacher")


def sources():
    paths = [Path("src/dcl/integration/models.py"), Path("src/dcl/integration/study.py"),
             *Path("src/dcl/multiteacher").glob("*.py"),
             *Path("src/dcl/multiteacher_e1").glob("*.py"),
             *Path("src/dcl/universal").glob("*.py"), Path("src/dcl/paperstudy/models.py"),
             *[Path("src/dcl/study") / p for p in ("config.py", "graphs.py", "storage.py")],
             Path("src/dcl/graphs.py"), Path("configs/integration-balanced.json")]
    return {str(p): file_hash(p) for p in sorted(paths)}


def setup(config):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)


def model_for(teacher, architecture, seed):
    torch.manual_seed(named_seed("integration-initialization", seed) % 2**63)
    return RootedPalettePolicy(teacher, architecture)


def cost_design(config, root):
    """Timing only: no accuracy/loss-based selection, no final training examples."""
    setup(config); root = Path(root); root.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = root / "design.json"
    if path.exists():
        d = json.loads(path.read_text())
        if d["config"] != config or d["source_hashes"] != sources():
            raise RuntimeError("Existing design config/source differs")
        return d
    rows = []
    for teacher in config["teachers"]:
        p = ColoringProgram(nx.cycle_graph(8), teacher, named_seed("integration-cost", teacher))
        state = p.snapshot(); choice = teacher_choices(state, p.neighbors, teacher)
        roots = np.resize(np.flatnonzero(state.permission), config["batch_decisions"])
        arrays = encode_roots(state, p.neighbors, roots)
        labels = np.column_stack((choice.color[roots], choice.auxiliary[roots], choice.mark[roots],
                                  np.full(len(roots), state.phase)))
        for architecture in config["architectures"]:
            model = model_for(teacher, architecture, config["seeds"][0])
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            cf = np.random.default_rng(named_seed("integration-cost-cf", teacher, architecture))
            # The measured own/CF regime is more expensive than teacher warmup.
            for _ in range(2):
                optimizer.zero_grad(); loss, _ = loss_one(model, arrays, labels, False, cf)
                loss.backward(); optimizer.step()
            tick = time.perf_counter()
            for _ in range(8):
                optimizer.zero_grad(); loss, _ = loss_one(model, arrays, labels, False, cf)
                loss.backward(); optimizer.step()
            seconds = (time.perf_counter() - tick) / 8
            rows.append(dict(teacher=teacher, architecture=architecture,
                             seconds_per_update=seconds, parameters=sum(p.numel() for p in model.parameters())))
    estimates = []
    for updates in config["candidate_update_budgets"]:
        seconds = config["cost_safety_factor"] * updates * len(config["seeds"]) * sum(r["seconds_per_update"] for r in rows)
        estimates.append(dict(updates=updates, training_seconds=seconds,
                              total_seconds=seconds + config["evaluation_reserve_seconds"]))
    feasible = [r for r in estimates if r["total_seconds"] <= config["wall_budget_seconds"]]
    if not feasible:
        raise RuntimeError("Even the smallest balanced design exceeds the declared cost budget")
    design = dict(config=config, updates=feasible[0]["updates"], estimates=estimates, timing=rows,
                  source_hashes=sources(), code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  python=platform.python_version(), device="cpu",
                  selected_without_scientific_outcomes=True,
                  note="Timing uses separate cycle8 states. First Linial phase conservatively represents BE timing. Training/validation/test graphs are not generated by this timing pilot.")
    atomic_json(path, design); atomic_json(OUT / "design.json", design)
    print(json.dumps({k: design[k] for k in ("updates", "estimates", "selected_without_scientific_outcomes")}), flush=True)
    return design


def train_cell(config, design, root, teacher, architecture, seed, pool, validation, deadline):
    directory = root / "runs" / teacher.lower() / architecture / f"s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "final.pt"
    if final.exists():
        saved = torch.load(final, map_location="cpu", weights_only=False)
        if saved["source_hashes"] != design["source_hashes"] or saved["updates"] != design["updates"]:
            raise RuntimeError("Existing final checkpoint differs")
        return {**saved["record"], "checkpoint_sha256": file_hash(final)}
    model = model_for(teacher, architecture, seed)
    initial_hash = weight_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    order = np.random.default_rng(named_seed("integration-minibatches", teacher, seed))
    cf = np.random.default_rng(named_seed("integration-counterfactual", teacher, seed))
    history = []; start = 0
    data_hash = hashlib.sha256(json.dumps([pool.manifest, validation.manifest], sort_keys=True).encode()).hexdigest()
    last = directory / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["source_hashes"] != design["source_hashes"] or saved["data_hash"] != data_hash or saved["initial_hash"] != initial_hash:
            raise RuntimeError("Exact resume requires unchanged source, data and initialization")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        order.bit_generator.state = saved["order_rng"]; cf.bit_generator.state = saved["cf_rng"]
        start, history = saved["updates"], saved["history"]
    tick = time.perf_counter(); loss_sum = 0.; batches = 0
    for update in range(start + 1, design["updates"] + 1):
        if time.perf_counter() > deadline:
            raise TimeoutError("Declared CPU study budget reached; retain completed cells and resume explicitly")
        model.train(); optimizer.zero_grad()
        positions = order.integers(len(pool.roots), size=config["batch_decisions"])
        arrays, labels = pool.batch(positions)
        loss = 0.
        for phase, part, truth in phase_groups(arrays, labels):
            local, _ = loss_one(model, part, truth, update <= max(1, round(design["updates"] * config["warmup_fraction"])), cf)
            loss = loss + local * len(truth) / len(labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite training loss")
        loss.backward(); optimizer.step()
        loss_sum += float(loss.detach()); batches += 1
        if update % 100 == 0 or update == design["updates"]:
            history.append(dict(update=update, mean_loss=loss_sum / batches,
                                segment_seconds=time.perf_counter() - tick))
            save_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                order_rng=order.bit_generator.state, cf_rng=cf.bit_generator.state, updates=update,
                history=history, initial_hash=initial_hash, data_hash=data_hash, source_hashes=design["source_hashes"]))
            tick = time.perf_counter(); loss_sum = 0.; batches = 0
    local = validate_local(model, validation, config["batch_decisions"])
    record = dict(teacher=teacher, architecture=architecture, seed=seed, updates=design["updates"],
                  supervised_root_decisions=design["updates"] * config["batch_decisions"],
                  train_root_pool=len(pool.roots), validation_root_pool=len(validation.roots),
                  parameters=sum(p.numel() for p in model.parameters()), initial_hash=initial_hash,
                  weights_hash=weight_hash(model), data_hash=data_hash, local=local,
                  training_seconds=sum(r["segment_seconds"] for r in history),
                  checkpoint=str(final), history=history)
    save_checkpoint(final, dict(model=model.state_dict(), record=record, source_hashes=design["source_hashes"],
                               updates=design["updates"], teacher=teacher, architecture=architecture, seed=seed))
    record["checkpoint_sha256"] = file_hash(final)
    atomic_json(directory / "record.json", record)
    print(f"complete {teacher}/{architecture}/s{seed}: updates={design['updates']} seconds={record['training_seconds']:.1f}", flush=True)
    return record


def evaluation_graphs(config, excluded):
    result = []
    for i, (n, family) in enumerate(config["graphs_for_evaluation"]):
        attempt = 0
        while True:
            seed = named_seed(config["protocol"], config["namespace"], "evaluation-graph", i, attempt)
            g = nx.complete_graph(4) if family == "CLIQUE" else generate_graph(n, 3, family, seed)
            fp = fingerprint(g)
            if fp not in excluded:
                break
            attempt += 1
        excluded.add(fp)
        result.append((g, dict(id=f"{family.lower()}-n{n}", n=n, family=family, graph_seed=seed,
                               fingerprint=fp, edges=list(map(list, g.edges())),
                               degrees=[g.degree(v) for v in range(n)], palette="global K=4")))
    return result


def sample_native(config, root, model, teacher, label, graph, metadata, deadline):
    path = root / "banks" / teacher.lower() / f"{metadata['id']}__{label}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    seeds = [named_seed(config["protocol"], config["namespace"], "new-independent-bank", teacher,
                        label, metadata["id"], j) for j in range(config["samples_per_bank"])]
    results = []
    for lo in range(0, len(seeds), 32):
        if time.perf_counter() > deadline:
            raise TimeoutError("Declared CPU budget reached during evaluation")
        cases = [(graph, dict(tape_seed=s)) for s in seeds[lo:lo+32]]
        results.extend(rollout_many(model, teacher, cases, 512))
    colors = np.asarray([r["colors"] for r in results], np.int16)
    # Same four-state vocabulary as the original local-palette studies.
    status = np.asarray([0 if r["success"] else (1 if r["failure"] in ("collision", "palette") else 2)
                         for r in results], np.int8)
    arrays = dict(colors=colors, status=status, rounds=np.asarray([r["native_steps"] for r in results], np.int16),
                  failure=np.asarray([r["failure"] or "" for r in results]), seeds=np.asarray(seeds, np.uint64))
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return arrays


def evaluate(config, design, root, records, excluded, deadline):
    graphs = evaluation_graphs(config, excluded)
    atomic_json(OUT / "graphs.json", [m for _, m in graphs])
    frozen = [dict(teacher=r["teacher"], architecture=r["architecture"], seed=r["seed"],
                   checkpoint=r["checkpoint"], sha256=file_hash(r["checkpoint"])) for r in records]
    atomic_json(OUT / "frozen-checkpoints.json", dict(models=frozen, source_hashes=design["source_hashes"],
        selection=config["selection"], generated_before_evaluation_banks=True))
    rows = []
    for teacher in config["teachers"]:
        actors = [(None, "exact", None)] + [(r, f"{r['architecture']}-s{r['seed']}", r["architecture"])
                                            for r in records if r["teacher"] == teacher]
        for record, label, architecture in actors:
            model = None
            if record is not None:
                model = model_for(teacher, architecture, record["seed"])
                model.load_state_dict(torch.load(record["checkpoint"], map_location="cpu", weights_only=False)["model"])
                model.eval()
            for graph, metadata in graphs:
                a = sample_native(config, root, model, teacher, label, graph, metadata, deadline)
                successes = a["status"] == 0
                assert all(all(x[u] != x[v] for u, v in graph.edges()) and np.all((x >= 0) & (x < 4))
                           for x in a["colors"][successes])
                path = root / "banks" / teacher.lower() / f"{metadata['id']}__{label}.npz"
                rows.append(dict(teacher=teacher, architecture=architecture or "exact", seed=None if record is None else record["seed"],
                    graph=metadata["id"], label=label, n=len(graph), samples=len(successes), successes=int(successes.sum()),
                    invalid=int((a["status"] == 1).sum()), cutoff=int((a["status"] == 2).sum()),
                    mean_native_steps=float(a["rounds"].mean()), bank=str(path), sha256=file_hash(path)))
            write_csv(OUT / "functional.csv", rows)
            print(f"evaluated {teacher}/{label} on {len(graphs)} fresh graphs", flush=True)
    atomic_json(OUT / "functional.json", rows)
    return rows


def run(config, root):
    setup(config); root = Path(root); design = cost_design(config, root)
    if design["source_hashes"] != sources():
        raise RuntimeError("Source changed after timing design")
    started = time.perf_counter(); deadline = started + config["wall_budget_seconds"]
    records, excluded, pools = [], set(), []
    try:
        for teacher in config["teachers"]:
            train = build_pool(config, root, teacher, "train")
            validation = build_pool(config, root, teacher, "validation")
            for pool in (train, validation):
                excluded.update(r["fingerprint"] for r in pool.manifest["records"])
                pools.append(dict(teacher=teacher, split=pool.manifest["config"]["split"], manifest=pool.manifest))
            for seed in config["seeds"]:
                for architecture in config["architectures"]:
                    records.append(train_cell(config, design, root, teacher, architecture, seed, train, validation, deadline))
                    atomic_json(OUT / "training.json", records)
            del train, validation
        atomic_json(OUT / "pool-manifests.json", pools)
        rows = evaluate(config, design, root, records, excluded, deadline)
        atomic_json(OUT / "execution.json", dict(complete=True, models=len(records), banks=len(rows),
                    wall_seconds=time.perf_counter()-started, device="cpu", sources_verified=sources()==design["source_hashes"]))
    except BaseException as exc:
        atomic_json(OUT / "execution.json", dict(complete=False, models=len(records),
                    wall_seconds=time.perf_counter()-started, error=repr(exc), completed_models_retained=True))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("cost", "run"))
    p.add_argument("--config", default="configs/integration-balanced.json")
    p.add_argument("--root", default="data/integration-balanced")
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    (cost_design if args.stage == "cost" else run)(config, args.root)
