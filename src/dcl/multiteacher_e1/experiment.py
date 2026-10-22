"""Budget-matched E1 follow-up to the frozen native-policy-v1 comparison."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
from threading import Lock
import time

import numpy as np
import torch

from dcl.multiteacher.data import load_pool
from dcl.multiteacher.experiment import final_cases, summarize, validation_cases, weight_hash
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import (Choices, ColoringProgram, DETECT, LINIAL, TEACHERS,
                                      named_seed, teacher_choices)
from dcl.study.config import digest
from dcl.study.graphs import fingerprint
from dcl.study.storage import atomic_json, file_hash, now, save_checkpoint
from dcl.universal.config import model_seed
from .model import RootedE1, native_prediction, parameters
from .supervision import (commit_targets, labels_from_inputs, loss_one, phase_groups,
                          tensor_arrays)

PROTOCOL = "e1-native-v1"
REPO = Path(__file__).resolve().parents[3]
INITIALIZATION_LOCK = Lock()


def metadata():
    sources = list((REPO / "src/dcl/universal").glob("*.py"))
    sources += list((REPO / "src/dcl/multiteacher").glob("*.py"))
    sources += list((REPO / "src/dcl/multiteacher_e1").glob("*.py"))
    sources += [REPO / p for p in ("src/dcl/study/config.py", "src/dcl/study/graphs.py",
                "src/dcl/study/storage.py", "experiments/run_multiteacher_e1.py",
                "experiments/supervise_multiteacher_e1.py", "experiments/analyze_multiteacher_e1.py",
                "experiments/analyze_multiteacher.py",
                "configs/multiteacher_e1/comparison.json", "docs/MULTITEACHER_E1.md",
                "experiments/rooted_multiteacher_cache.py")]
    return {"source_hashes": {str(p.relative_to(REPO)): file_hash(p) for p in sorted(sources)},
            "versions": {n: version(n) for n in ("torch", "numpy", "networkx", "scipy", "matplotlib")},
            "python": platform.python_version(), "device": "cpu", "created_utc": now(),
            "git_commit": subprocess.check_output(["git", "-c", f"safe.directory={REPO}",
                                                    "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}


def verify_metadata(expected):
    current = metadata()
    for field in ("source_hashes", "versions", "python", "device"):
        if current[field] != expected[field]:
            raise RuntimeError(f"Experiment dependency changed: {field}")


def initialize(config, teacher, seed):
    # Linear/ReLU/MAX modules have no stochastic training operations. Serialize
    # global seeding and model construction; minibatches/counterfactuals are private.
    with INITIALIZATION_LOCK:
        value = model_seed(seed, "initialization")
        random.seed(value); np.random.seed(value % 2**32); torch.manual_seed(value)
        model = RootedE1(teacher, config["hidden"], config["message"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    order = np.random.default_rng(named_seed(PROTOCOL, "minibatches", seed))
    cf = np.random.default_rng(named_seed(PROTOCOL, "counterfactual", seed))
    return model, optimizer, order, cf


def predictions(model, arrays, phases):
    n = len(phases)
    result = [np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)]
    for phase in np.unique(phases):
        use = phases == phase
        out = model(*tensor_arrays(tuple(a[use] for a in arrays)), phase=int(phase))
        for target, pred in zip(result, native_prediction(out)):
            target[use] = pred.detach().numpy()
    return tuple(result)


@torch.inference_mode()
def rollout_many(model, teacher, cases, batch_size=1680):
    programs = [ColoringProgram(g, teacher, m["tape_seed"]) for g, m in cases]
    counts = np.zeros(len(cases), dtype=np.int64)
    tick = time.perf_counter()
    if model is not None:
        model.eval()
    while any(not p.finished for p in programs):
        pending = [(i, p, p.snapshot(copy_arrays=False)) for i, p in enumerate(programs) if not p.finished]
        choices = {i: Choices(np.zeros(p.n, np.int64), np.zeros(p.n, np.int64), np.zeros(p.n, bool))
                   for i, p, _ in pending}
        buffer, destinations, phases, buffered = [], [], [], 0
        def flush():
            nonlocal buffer, destinations, phases, buffered
            if not buffered:
                return
            arrays = tuple(np.concatenate([a[j] for a in buffer]) for j in range(4))
            pred = predictions(model, arrays, np.concatenate(phases))
            cursor = 0
            for i, roots in destinations:
                take = slice(cursor, cursor + len(roots))
                for target, values in zip((choices[i].color, choices[i].auxiliary, choices[i].mark), pred):
                    target[roots] = values[take]
                cursor += len(roots)
            buffer, destinations, phases, buffered = [], [], [], 0
        for i, program, state in pending:
            roots = np.flatnonzero(state.permission)
            counts[i] += len(roots)
            if model is None:
                choices[i] = teacher_choices(state, program.neighbors, teacher)
                continue
            cursor = 0
            while cursor < len(roots):
                part = roots[cursor:cursor + batch_size - buffered]
                buffer.append(encode_roots(state, program.neighbors, part))
                phases.append(np.full(len(part), state.phase, dtype=np.int64))
                destinations.append((i, part))
                cursor += len(part); buffered += len(part)
                if buffered == batch_size:
                    flush()
        if model is not None:
            flush()
        for i, program, state in pending:
            program.apply(state, choices[i])
    seconds = time.perf_counter() - tick
    return [{"success": p.failure is None, "failure": p.failure, "native_steps": p.steps,
             "decision_nodes": int(counts[i]), "monochromatic_edges": int(p.bad_edges()),
             "out_of_palette": int(((p.colors < -1) | (p.colors >= 4)).sum()),
             "uncolored_nodes": int((p.colors < 0).sum()), "colors": p.colors.tolist(),
             "cohort_seconds": seconds, "cohort_size": len(programs)} for i, p in enumerate(programs)]


@torch.inference_mode()
def validate_local(model, pool, batch_size):
    model.eval()
    totals, losses = defaultdict(int), 0.
    for start in range(0, len(pool.roots), batch_size):
        arrays, labels = pool.batch(np.arange(start, min(start + batch_size, len(pool.roots))))
        for phase, part, truth in phase_groups(arrays, labels):
            loss, out = loss_one(model, part, truth, True, None)
            pred = [p.numpy() for p in native_prediction(out)]
            field = 2 if phase == DETECT else (1 if phase == LINIAL else 0)
            correct = int((pred[field] == truth[:, field]).sum())
            n = len(truth)
            totals["decisions"] += n; totals["correct"] += correct
            totals[f"phase_{phase}_decisions"] += n; totals[f"phase_{phase}_correct"] += correct
            losses += float(loss) * n
            if phase == DETECT:
                totals["mark_tp"] += int((pred[2] & (truth[:, 2] > 0)).sum())
                totals["mark_fp"] += int((pred[2] & (truth[:, 2] == 0)).sum())
                totals["mark_fn"] += int((~pred[2] & (truth[:, 2] > 0)).sum())
            elif phase != LINIAL:
                target = labels_from_inputs(part, truth, model.teacher)
                own_p, own_c = out.p.numpy(), out.candidates.numpy()
                own_truth = commit_targets(target, own_p, own_c)
                totals["component_decisions"] += n
                totals["participation_correct"] += int((own_p[:, 0] == target.participate[:, 0]).sum())
                totals["candidate_correct"] += int((own_c[:, 0] == target.candidate[:, 0]).sum())
                totals["own_commit_correct"] += int(((out.commit.numpy() > 0) == own_truth).sum())
                totals["unsafe_own_commits"] += int(((out.commit.numpy() > 0) & ~own_truth).sum())
    return {**totals, "accuracy": totals["correct"] / totals["decisions"], "loss": losses / totals["decisions"]}


def train_one(config, root, teacher, seed, pool, validation, provenance, stop_after=None):
    directory = Path(root) / "runs" / teacher.lower() / f"s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    model, optimizer, order, cf = initialize(config, teacher, seed)
    initial, core_initial = weight_hash(model), weight_hash(model.core)
    data_hash = digest([pool.manifest, validation.manifest])
    history, start, elapsed = [], 0, 0.
    last = directory / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        verify_metadata(saved["provenance"])
        if (saved["config"], saved["data_hash"], saved["initial_state_hash"]) != (config, data_hash, initial):
            raise RuntimeError("Resume configuration/data/initialization differs")
        model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer_state_dict"])
        order.bit_generator.state, cf.bit_generator.state = saved["order_rng"], saved["counterfactual_rng"]
        history, start, elapsed = saved["history"], saved["updates"], saved["elapsed_seconds"]
        provenance = saved["provenance"]
        if start in config["checkpoints"] and not (directory / f"step-{start:06d}.pt").exists():
            save_checkpoint(directory / f"step-{start:06d}.pt", saved)
    print(f"train E1 {teacher} s{seed} resume={start} parameters={parameters(model)}", flush=True)
    tick, loss_sum, correct_sum, since_log = time.perf_counter(), 0., 0, 0
    end = min(config["updates"], stop_after) if stop_after is not None else config["updates"]
    for step in range(start + 1, end + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        positions = order.integers(len(pool.roots), size=config["batch_decisions"])
        arrays, labels = pool.batch(positions)
        total = 0
        for phase, part, truth in phase_groups(arrays, labels):
            loss, output = loss_one(model, part, truth, step <= config["warmup_updates"], cf)
            total = total + loss * (len(truth) / len(labels))
            field = 2 if phase == DETECT else (1 if phase == LINIAL else 0)
            pred = native_prediction(output)[field].detach().numpy()
            correct_sum += int((pred == truth[:, field]).sum())
        if not bool(torch.isfinite(total)):
            raise RuntimeError("Nonfinite E1 native loss")
        total.backward(); optimizer.step()
        loss_sum += float(total.detach()); since_log += 1
        if step % config["log_every"] == 0 or step in config["checkpoints"] or step == end:
            elapsed += time.perf_counter() - tick
            local, rollouts = None, None
            if step in config["checkpoints"]:
                local = validate_local(model, validation, config["batch_decisions"])
                rows = rollout_many(model, teacher, validation_cases(config), config["batch_decisions"])
                rollouts = {"successes": sum(r["success"] for r in rows), "trials": len(rows),
                            "success_rate": float(np.mean([r["success"] for r in rows])),
                            "failures": [r["failure"] for r in rows]}
            row = {"updates": step, "decision_nodes_seen": step * config["batch_decisions"],
                   "train_loss": loss_sum / since_log,
                   "train_accuracy": correct_sum / (since_log * config["batch_decisions"]),
                   "validation_local": local, "validation_rollouts": rollouts, "elapsed_seconds": elapsed}
            history.append(row)
            saved = {"config": config, "teacher": teacher, "seed": seed, "updates": step,
                     "decision_nodes_seen": row["decision_nodes_seen"], "initial_state_hash": initial,
                     "initial_core_hash": core_initial, "data_hash": data_hash, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "order_rng": order.bit_generator.state,
                     "counterfactual_rng": cf.bit_generator.state, "history": history, "elapsed_seconds": elapsed,
                     "provenance": provenance, "parameters": parameters(model)}
            save_checkpoint(last, saved)
            if step in config["checkpoints"]:
                save_checkpoint(directory / f"step-{step:06d}.pt", saved)
            atomic_json(directory / "history.json", history)
            print(f"E1 {teacher} s{seed} step={step} loss={row['train_loss']:.5f} "
                  f"accuracy={row['train_accuracy']:.5f} val={None if local is None else round(local['accuracy'],5)} "
                  f"seconds={elapsed:.1f}", flush=True)
            tick, loss_sum, correct_sum, since_log = time.perf_counter(), 0., 0, 0
    return directory / f"step-{config['updates']:06d}.pt"


def freeze(config, root, provenance):
    models, initial = [], {}
    for teacher in TEACHERS:
        for seed in config["seeds"]:
            relative = Path("runs") / teacher.lower() / f"s{seed}" / f"step-{config['updates']:06d}.pt"
            saved = torch.load(Path(root) / relative, map_location="cpu", weights_only=False)
            if saved["updates"] != config["updates"] or saved["config"] != config:
                raise RuntimeError("Cannot freeze incomplete/different run")
            verify_metadata(saved["provenance"])
            if seed in initial and initial[seed] != saved["initial_core_hash"]:
                raise RuntimeError("E1 core initializations are not paired")
            initial[seed] = saved["initial_core_hash"]
            models.append({"teacher": teacher, "seed": seed, "path": str(relative),
                           "sha256": file_hash(Path(root) / relative), "updates": saved["updates"],
                           "decisions": saved["decision_nodes_seen"], "parameters": saved["parameters"],
                           "initial_core_hash": saved["initial_core_hash"]})
    value = {"protocol": PROTOCOL, "config": config, "provenance": provenance, "models": models,
             "created_utc": now(), "selection": "last prespecified update; validation for monitoring only"}
    path = Path(root) / "frozen.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["models"] != models or previous["config"] != config:
            raise RuntimeError("Frozen experiment changed")
        return previous
    verify_metadata(provenance)
    atomic_json(path, value)
    return value


def evaluate(config, root, frozen, excluded):
    verify_metadata(frozen["provenance"])
    cases = list(final_cases(config, excluded))
    rows = []
    for teacher in TEACHERS:
        items = [(None, None)] + [(m["seed"], m) for m in frozen["models"] if m["teacher"] == teacher]
        for seed, item in items:
            model = None
            if item:
                path = Path(root) / item["path"]
                if file_hash(path) != item["sha256"]:
                    raise RuntimeError("Frozen model changed")
                saved = torch.load(path, map_location="cpu", weights_only=False)
                model = RootedE1(teacher, config["hidden"], config["message"])
                model.load_state_dict(saved["model_state_dict"])
            actor = "exact" if seed is None else f"s{seed}"
            folder = Path(root) / "evaluations" / teacher.lower() / actor
            folder.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(cases), config["cohort_batch"]):
                cohort = cases[start:start + config["cohort_batch"]]
                missing = [(g, m) for g, m in cohort if not (folder / f"{m['id']}.json").exists()]
                if missing:
                    results = rollout_many(model, teacher, missing, config["batch_decisions"])
                    for (_, case), result in zip(missing, results):
                        if seed is None and not result["success"]:
                            raise RuntimeError("Exact reference failed")
                        atomic_json(folder / f"{case['id']}.json", {"freeze_hash": digest(frozen), "case": case,
                                    "teacher": teacher, "seed": seed, "result": result})
                for graph, case in cohort:
                    record = json.loads((folder / f"{case['id']}.json").read_text())
                    if record["freeze_hash"] != digest(frozen) or record["case"] != case:
                        raise RuntimeError("Persisted evaluation differs")
                    result, colors = record["result"], np.array(record["result"]["colors"])
                    if colors.shape != (len(graph),):
                        raise RuntimeError("Invalid coloring shape")
                    bad = sum(colors[u] >= 0 and colors[u] == colors[v] for u, v in graph.edges())
                    good = bool(np.all((colors >= 0) & (colors < 4)) and bad == 0)
                    if good != result["success"] or bad != result["monochromatic_edges"]:
                        raise RuntimeError("Coloring disagrees with recorded correctness")
                    rows.append({**case, "teacher": teacher, "seed": seed,
                                 **{k: v for k, v in result.items() if k != "colors"}})
                if start % (4 * config["cohort_batch"]) == 0:
                    print(f"evaluate E1 {teacher} {actor} {start + len(cohort)}/{len(cases)}", flush=True)
    atomic_json(Path(root) / "evaluation-rows.json", rows)
    return rows


def prepare(config, root, source):
    if config["protocol"] != PROTOCOL or tuple(config["teachers"]) != TEACHERS:
        raise ValueError("Unknown protocol/teacher set")
    root = Path(root); source = Path(source)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != config:
        raise RuntimeError("Existing plan differs")
    atomic_json(plan_path, config)
    info = metadata()
    provenance_path = root / "provenance.json"
    if provenance_path.exists():
        info = json.loads(provenance_path.read_text()); verify_metadata(info)
    else:
        atomic_json(provenance_path, info)
    for relative, expected in info["source_hashes"].items():
        target = root / "sources" / relative
        if target.exists() and file_hash(target) != expected:
            raise RuntimeError("Archived source changed")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(REPO / relative, target)
    data, excluded, common = {}, set(), {}
    fingerprints_by_split = defaultdict(set)
    manifests = {}
    for teacher in TEACHERS:
        manifests[teacher] = {}
        for split in ("train", "validation"):
            folder = source / "pools" / teacher.lower() / split
            manifest = json.loads((folder / "manifest.json").read_text())
            expected = {"protocol": config["data_protocol"], "namespace": config["namespace"], "teacher": teacher,
                        "split": split, "n": config["n"], "graphs": config["graphs"] if split == "train" else config["val_graphs"],
                        "tapes": config["tapes"] if split == "train" else 1}
            if manifest["config"] != expected:
                raise RuntimeError("Source pool differs from prespecified data")
            for name, sha in manifest["hashes"].items():
                if file_hash(folder / manifest["attempt"] / name) != sha:
                    raise RuntimeError("Immutable source pool changed")
            signatures = digest([[r["fingerprint"], [t["tape_seed"] for t in r["trajectories"]]]
                                 for r in manifest["records"]])
            if split in common and common[split] != signatures:
                raise RuntimeError("Teachers do not share source graphs/tapes")
            common[split] = signatures
            excluded.update(r["fingerprint"] for r in manifest["records"])
            fingerprints_by_split[split].update(r["fingerprint"] for r in manifest["records"])
            data[teacher, split] = (folder, manifest)
            manifests[teacher][split] = {"manifest_sha256": file_hash(folder / "manifest.json"),
                                       "source_manifest_digest": digest(manifest), "decisions": manifest["decisions"]}
    if fingerprints_by_split["train"] & fingerprints_by_split["validation"]:
        raise RuntimeError("Training/validation graphs overlap")
    excluded.update(fingerprint(g) for g, _ in validation_cases(config))
    atomic_json(root / "source-pools.json", {"source_root": str(source.resolve()), "manifests": manifests,
                                           "paired_signatures": common})
    return info, data, excluded


def run(config, root, source, trainers=1, stage="all", resident_factory=None):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    root, source = Path(root), Path(source)
    info, data, excluded = prepare(config, root, source)
    atomic_json(root / "execution.json", {"created_utc": now(), "trainers": trainers,
                "processes": 1, "torch_threads": config["threads"], "device": "cpu",
                "grouped_by_teacher": True, "rng": "private minibatch/CF; serialized initialization"})
    if stage in ("train", "all"):
        for teacher in TEACHERS:
            pool = load_pool(*data[teacher, "train"])
            validation = load_pool(*data[teacher, "validation"])
            if teacher == "GREEDY" and resident_factory is not None:
                pool = resident_factory(pool, source / "caches/greedy-roots")
            with ThreadPoolExecutor(max_workers=trainers) as executor:
                futures = [executor.submit(train_one, config, root, teacher, seed, pool, validation, info)
                           for seed in config["seeds"]]
                for future in futures:
                    future.result()
            del pool, validation
    if stage in ("evaluate", "all"):
        frozen = freeze(config, root, info)
        rows = evaluate(config, root, frozen, excluded)
        summary = summarize(config, root, rows)
        verify_metadata(info)
        for item in frozen["models"]:
            if file_hash(root / item["path"]) != item["sha256"]:
                raise RuntimeError("Weights changed after freeze")
        atomic_json(root / "complete.json", {"status": "complete", "created_utc": now(),
                    "models": len(frozen["models"]), "cases": summary["cases"],
                    "evaluations": summary["evaluation_rows"], "frozen_sha256": file_hash(root / "frozen.json")})
        return summary
