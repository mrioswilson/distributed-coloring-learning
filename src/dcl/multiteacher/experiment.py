"""Reproducible, budget-matched comparison of seven native local policies."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import multiprocessing

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from dcl.study.config import digest
from dcl.study.graphs import generate_graph, fingerprint, degree_stats
from dcl.study.storage import atomic_json, file_hash, now, provenance, save_checkpoint
from .data import build_pool, load_pool
from .model import LocalGNN, encode_roots, tensor_batch
from .programs import ColoringProgram, Choices, TEACHERS, LINIAL, DETECT, named_seed, teacher_choices

PROTOCOL = "native-policy-v1"


def check_config(config):
    if config["protocol"] != PROTOCOL or tuple(config["teachers"]) != TEACHERS:
        raise ValueError("Unknown protocol/teacher set")
    for key in ("graphs", "tapes", "val_graphs", "updates", "batch_decisions", "log_every", "threads", "workers", "test_tapes", "cohort_batch"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["n"] < 4 or config["n"] % 2 or not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Invalid graph size or initialization seeds")
    if max(config["checkpoints"]) != config["updates"] or min(config["checkpoints"]) < 1:
        raise ValueError("Checkpoints must end at the prespecified update budget")


def source_metadata():
    metadata = provenance()
    metadata["source_hashes"]["experiments/run_multiteacher.py"] = file_hash("experiments/run_multiteacher.py")
    return metadata


def verify_runtime(expected):
    current = source_metadata()
    if current["source_hashes"] != expected["source_hashes"] or current["versions"] != expected["versions"]:
        raise RuntimeError("Sources/runtime changed; resume refused")


def weight_hash(model):
    value = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        value.update(name.encode()); value.update(parameter.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def initialize(config, seed):
    value = named_seed(PROTOCOL, config["namespace"], "initialization", seed)
    random.seed(value); np.random.seed(value % 2**32); torch.manual_seed(value)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    model = LocalGNN(config["hidden"], config["message"])
    return model, torch.optim.Adam(model.parameters(), lr=config["lr"]), np.random.default_rng(
        named_seed(PROTOCOL, config["namespace"], "minibatches", seed))


def policy_loss(outputs, labels):
    color, auxiliary, mark = outputs
    labels = torch.from_numpy(labels)
    phase = labels[:, 3]
    a, m = phase == LINIAL, phase == DETECT
    c = ~(a | m)
    total = color.sum() * 0
    correct = 0
    for use, logits, target in ((c, color, labels[:, 0]), (a, auxiliary, labels[:, 1])):
        if bool(use.any()):
            total = total + F.cross_entropy(logits[use], target[use], reduction="sum")
            correct += int((logits[use].detach().argmax(1) == target[use]).sum())
    if bool(m.any()):
        total = total + F.binary_cross_entropy_with_logits(mark[m], labels[m, 2].float(), reduction="sum")
        correct += int(((mark[m].detach() > 0) == (labels[m, 2] > 0)).sum())
    return total / len(labels), correct


@torch.inference_mode()
def validate_local(model, pool, batch_size):
    model.eval()
    totals = defaultdict(int)
    loss_sum = 0.0
    for start in range(0, len(pool.roots), batch_size):
        arrays, labels = pool.batch(np.arange(start, min(start + batch_size, len(pool.roots))))
        outputs = model(*tensor_batch(arrays))
        loss, correct = policy_loss(outputs, labels)
        loss_sum += float(loss) * len(labels)
        totals["correct"] += correct; totals["decisions"] += len(labels)
        color, auxiliary, mark = (v.cpu().numpy() for v in outputs)
        for phase in np.unique(labels[:, 3]):
            use = labels[:, 3] == phase
            pred = ((mark > 0).astype(int) if phase == DETECT else
                    (auxiliary.argmax(1) if phase == LINIAL else color.argmax(1)))
            target = labels[:, 2 if phase == DETECT else (1 if phase == LINIAL else 0)]
            totals[f"phase_{phase}_decisions"] += int(use.sum())
            totals[f"phase_{phase}_correct"] += int((pred[use] == target[use]).sum())
            if phase == DETECT:
                totals["mark_tp"] += int((use & (pred == 1) & (target == 1)).sum())
                totals["mark_fp"] += int((use & (pred == 1) & (target == 0)).sum())
                totals["mark_fn"] += int((use & (pred == 0) & (target == 1)).sum())
    return {**dict(totals), "accuracy": totals["correct"] / totals["decisions"],
            "loss": loss_sum / totals["decisions"]}


@torch.inference_mode()
def rollout_many(model, teacher, cases, batch_size=1680):
    """Interleave independent graphs, preserving simultaneous actions within each.

    This amortizes the overhead of single-vertex greedy/repair steps. No exact
    teacher is invoked when model is provided.
    """
    programs = [ColoringProgram(graph, teacher, meta["tape_seed"]) for graph, meta in cases]
    started = time.perf_counter()
    decision_counts = np.zeros(len(cases), np.int64)
    if model is not None:
        model.eval()
    while any(not p.finished for p in programs):
        pending = [(i, p, p.snapshot(copy_arrays=False)) for i, p in enumerate(programs) if not p.finished]
        choices = {i: Choices(np.zeros(p.n, np.int64), np.zeros(p.n, np.int64), np.zeros(p.n, bool)) for i, p, s in pending}
        arrays_buffer, destinations, buffered = [], [], 0
        def flush():
            nonlocal arrays_buffer, destinations, buffered
            if not buffered:
                return
            batch = tuple(np.concatenate([a[j] for a in arrays_buffer], axis=0) for j in range(4))
            outputs = model(*tensor_batch(batch))
            predictions = outputs[0].argmax(1).numpy(), outputs[1].argmax(1).numpy(), (outputs[2] > 0).numpy()
            cursor = 0
            for i, roots in destinations:
                selected = slice(cursor, cursor + len(roots))
                choices[i].color[roots] = predictions[0][selected]
                choices[i].auxiliary[roots] = predictions[1][selected]
                choices[i].mark[roots] = predictions[2][selected]
                cursor += len(roots)
            arrays_buffer, destinations, buffered = [], [], 0
        for i, program, state in pending:
            roots = np.flatnonzero(state.permission)
            decision_counts[i] += len(roots)
            if model is None:
                choices[i] = teacher_choices(state, program.neighbors, teacher)
                continue
            cursor = 0
            while cursor < len(roots):
                part = roots[cursor:cursor + batch_size - buffered]
                arrays_buffer.append(encode_roots(state, program.neighbors, part))
                destinations.append((i, part))
                buffered += len(part); cursor += len(part)
                if buffered == batch_size:
                    flush()
        if model is not None:
            flush()
        for i, program, state in pending:
            program.apply(state, choices[i])
    seconds = time.perf_counter() - started
    return [{"success": p.failure is None, "failure": p.failure, "native_steps": p.steps,
             "decision_nodes": int(decision_counts[i]), "monochromatic_edges": int(p.bad_edges()),
             "out_of_palette": int(((p.colors < -1) | (p.colors >= 4)).sum()),
             "uncolored_nodes": int((p.colors < 0).sum()), "colors": p.colors.tolist(),
             "cohort_seconds": seconds, "cohort_size": len(programs)} for i, p in enumerate(programs)]


def validation_cases(config):
    result = []
    for n in (config["n"], 4 * config["n"]):
        for family in ("REGULAR", "IRREGULAR"):
            seed = named_seed(PROTOCOL, config["namespace"], "validation_rollout", n, family)
            graph = generate_graph(n, 3, family, seed)
            result.append((graph, {"tape_seed": named_seed(seed, "tape"), "n": n, "family": family}))
    return result


def train_one(config, root, teacher, seed, pool, validation, metadata):
    directory = Path(root) / "runs" / teacher.lower() / f"s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"step-{config['updates']:06d}.pt"
    model, optimizer, order = initialize(config, seed)
    initial_hash = weight_hash(model)
    data_hash = digest([pool.manifest, validation.manifest])
    history, start_step, elapsed = [], 0, 0.0
    if (directory / "last.pt").exists():
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        verify_runtime(saved["provenance"])
        if saved["config"] != config or saved["data_hash"] != data_hash or saved["initial_state_hash"] != initial_hash:
            raise RuntimeError("Resume configuration/data/initialization differs")
        model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer_state_dict"])
        order.bit_generator.state = saved["order_rng"]
        torch.set_rng_state(saved["torch_rng"])
        history, start_step, elapsed = saved["history"], saved["updates"], saved["elapsed_seconds"]
        if start_step in config["checkpoints"] and not (directory / f"step-{start_step:06d}.pt").exists():
            save_checkpoint(directory / f"step-{start_step:06d}.pt", saved)
    print(f"train {teacher} seed={seed} resume={start_step} parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    if start_step == config["updates"]:
        return final
    tick, loss_sum, correct_sum, since_log = time.perf_counter(), 0., 0, 0
    for step in range(start_step + 1, config["updates"] + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        positions = order.integers(len(pool.roots), size=config["batch_decisions"])
        arrays, labels = pool.batch(positions)
        loss, correct = policy_loss(model(*tensor_batch(arrays)), labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite policy loss")
        loss.backward(); optimizer.step()
        loss_sum += float(loss.detach()); correct_sum += correct; since_log += 1
        if step % config["log_every"] == 0 or step in config["checkpoints"]:
            elapsed += time.perf_counter() - tick
            local, rollouts = None, None
            if step in config["checkpoints"]:
                local = validate_local(model, validation, config["batch_decisions"])
                runs = rollout_many(model, teacher, validation_cases(config), config["batch_decisions"])
                rollouts = {"successes": sum(r["success"] for r in runs), "trials": len(runs),
                            "success_rate": float(np.mean([r["success"] for r in runs])),
                            "failures": [r["failure"] for r in runs]}
            row = {"updates": step, "decision_nodes_seen": step * config["batch_decisions"],
                   "train_loss": loss_sum / since_log,
                   "train_accuracy": correct_sum / (since_log * config["batch_decisions"]),
                   "validation_local": local, "validation_rollouts": rollouts, "elapsed_seconds": elapsed}
            history.append(row)
            saved = {"config": config, "teacher": teacher, "seed": seed, "updates": step,
                     "decision_nodes_seen": row["decision_nodes_seen"], "initial_state_hash": initial_hash,
                     "data_hash": data_hash, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "order_rng": order.bit_generator.state,
                     "torch_rng": torch.get_rng_state(), "history": history, "elapsed_seconds": elapsed,
                     "provenance": metadata, "parameters": sum(p.numel() for p in model.parameters())}
            save_checkpoint(directory / "last.pt", saved)
            if step in config["checkpoints"]:
                save_checkpoint(directory / f"step-{step:06d}.pt", saved)
            atomic_json(directory / "history.json", history)
            print(f"{teacher} seed={seed} updates={step} loss={row['train_loss']:.5f} "
                  f"accuracy={row['train_accuracy']:.5f} val={None if local is None else round(local['accuracy'],5)} "
                  f"seconds={elapsed:.1f}", flush=True)
            tick, loss_sum, correct_sum, since_log = time.perf_counter(), 0., 0, 0
    return final


def train_worker(config, root, teacher, seed, train_manifest, val_manifest, metadata):
    """Independent RNG/optimizer process; data files are shared read-only maps."""
    folder = Path(root) / "pools" / teacher.lower()
    pool = load_pool(folder / "train", train_manifest)
    validation = load_pool(folder / "validation", val_manifest)
    return train_one(config, root, teacher, seed, pool, validation, metadata)


def freeze(config, root, paths, metadata):
    expected_pairs = {(t, s) for t in TEACHERS for s in config["seeds"]}
    frozen = {"protocol": PROTOCOL, "config": config, "provenance": metadata, "created_utc": now(),
              "selection": "last prespecified update; validation is monitoring only", "models": []}
    initial = {}
    for path in paths:
        saved = torch.load(path, weights_only=False, map_location="cpu")
        if saved["updates"] != config["updates"]:
            raise RuntimeError("Cannot freeze unfinished models")
        if saved["seed"] in initial and initial[saved["seed"]] != saved["initial_state_hash"]:
            raise RuntimeError("Initial weights are not matched across teachers")
        initial[saved["seed"]] = saved["initial_state_hash"]
        frozen["models"].append({"path": str(path.relative_to(root)), "teacher": saved["teacher"],
                                 "seed": saved["seed"], "sha256": file_hash(path),
                                 "updates": saved["updates"], "decisions": saved["decision_nodes_seen"],
                                 "initial_state_hash": saved["initial_state_hash"], "parameters": saved["parameters"]})
    if {(m["teacher"], m["seed"]) for m in frozen["models"]} != expected_pairs:
        raise RuntimeError("Incomplete teacher/initialization matrix")
    path = Path(root) / "frozen.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["models"] != frozen["models"] or old["config"] != config:
            raise RuntimeError("Frozen experiment differs")
        return old
    verify_runtime(metadata)
    atomic_json(path, frozen)
    return frozen


def final_cases(config, excluded):
    for n, count in config["test_counts"]:
        for family in ("REGULAR", "IRREGULAR", "PATH", "CYCLE"):
            for i in range(count if family in ("REGULAR", "IRREGULAR") else 1):
                graph_seed = named_seed(PROTOCOL, config["namespace"], "final_test", n, family, i)
                graph = generate_graph(n, 3, family, graph_seed)
                fp = fingerprint(graph)
                if fp in excluded:
                    raise RuntimeError("Final test overlaps source/validation data")
                for j in range(config["test_tapes"]):
                    yield graph, {"id": f"{family}-n{n}-g{i}-t{j}", "n": n, "family": family, "graph_i": i,
                                  "tape_i": j, "graph_seed": graph_seed, "fingerprint": fp,
                                  "tape_seed": named_seed(graph_seed, "tape", j), **degree_stats(graph)}
    graph = nx.complete_graph(4)
    for j in range(4):
        yield graph, {"id": f"CLIQUE-n4-g0-t{j}", "n": 4, "family": "CLIQUE", "graph_i": 0,
                      "tape_i": j, "graph_seed": None, "fingerprint": fingerprint(graph),
                      "tape_seed": named_seed(PROTOCOL, config["namespace"], "final_test", "clique", j), **degree_stats(graph)}


def evaluate(config, root, frozen, excluded):
    verify_runtime(frozen["provenance"])
    cases = list(final_cases(config, excluded))
    rows = []
    for teacher in TEACHERS:
        actors = [(None, None)] + [(m["seed"], m) for m in frozen["models"] if m["teacher"] == teacher]
        for seed, item in actors:
            model = None
            if item:
                path = Path(root) / item["path"]
                if file_hash(path) != item["sha256"]:
                    raise RuntimeError("Frozen model changed")
                saved = torch.load(path, weights_only=False, map_location="cpu")
                model = LocalGNN(config["hidden"], config["message"])
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
                            raise RuntimeError(f"Exact reference failed: {teacher}/{case['id']}")
                        atomic_json(folder / f"{case['id']}.json", {"freeze_hash": digest(frozen),
                            "case": case, "teacher": teacher, "seed": seed, "result": result})
                for graph, case in cohort:
                    record = json.loads((folder / f"{case['id']}.json").read_text())
                    if record["freeze_hash"] != digest(frozen) or record["case"] != case:
                        raise RuntimeError("Persisted evaluation differs")
                    result = record["result"]
                    colors = np.asarray(result["colors"])
                    if colors.shape != (len(graph),):
                        raise RuntimeError("Invalid coloring shape")
                    bad = int(sum(colors[u] >= 0 and colors[u] == colors[v] for u, v in graph.edges()))
                    success = bool(np.all((colors >= 0) & (colors < 4)) and not bad)
                    if success != result["success"] or bad != result["monochromatic_edges"]:
                        raise RuntimeError("Stored correctness disagrees with coloring")
                    rows.append({**case, "teacher": teacher, "seed": seed,
                                 **{k: v for k, v in result.items() if k != "colors"}})
                if start % (config["cohort_batch"] * 4) == 0:
                    print(f"evaluate {teacher} {actor} cases={min(start+len(cohort),len(cases))}/{len(cases)}", flush=True)
    atomic_json(Path(root) / "evaluation-rows.json", rows)
    return rows


def summarize(config, root, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["teacher"], row["seed"], row["family"], row["n"])].append(row)
    per_seed = []
    for (teacher, seed, family, n), values in groups.items():
        successes = [r for r in values if r["success"]]
        per_seed.append({"teacher": teacher, "seed": seed, "family": family, "n": n, "trials": len(values),
                         "successes": len(successes), "success_rate": len(successes) / len(values),
                         "collision_failures": sum(r["failure"] == "collision" for r in values),
                         "palette_failures": sum(r["failure"] == "palette" for r in values),
                         "incomplete_failures": sum(r["failure"] == "incomplete" for r in values),
                         "mean_steps_success": float(np.mean([r["native_steps"] for r in successes])) if successes else None})
    with (Path(root) / "success-by-seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0])); writer.writeheader(); writer.writerows(per_seed)
    cells = []
    for teacher in TEACHERS:
        for n, _ in config["test_counts"]:
            for family in ("REGULAR", "IRREGULAR", "PATH", "CYCLE"):
                rates = [r["success_rate"] for r in per_seed if r["teacher"] == teacher and r["seed"] is not None
                         and r["family"] == family and r["n"] == n]
                cells.append({"teacher": teacher, "n": n, "family": family, "rates_by_seed": rates,
                              "mean": float(np.mean(rates)), "min": min(rates), "max": max(rates)})
    summary = {"config": config, "cases": len({r["id"] for r in rows}), "evaluation_rows": len(rows),
               "per_seed": per_seed, "cells": cells, "scope": [
                   "Learned local choices inside native phase/clock controllers; arithmetic representations are supplied.",
                   "BE: maximum degree 3, Linial initialization and KW base of Delta-Color; no high-degree recursion.",
                   "Multi-Trials: repeated ColorTrials with global bound; no RankingTrials or fast-paper round claim.",
                   "Shared source graph/tape pool across all initialization replicas; no claim of exact M6B reproduction.",
                   "Five initializations are exploratory; zero observed errors do not imply universal correctness.",
                   "Native program steps of sequential and parallel algorithms are different cost models.",
                   "All lower-degree controls use four colors, not their smaller degree+1 palette."]}
    atomic_json(Path(root) / "summary.json", summary)
    learning_curves(config, root)
    return summary


def learning_curves(config, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = []
    for teacher in TEACHERS:
        for seed in config["seeds"]:
            path = Path(root) / "runs" / teacher.lower() / f"s{seed}" / "history.json"
            for row in json.loads(path.read_text()):
                if row["validation_local"] is not None:
                    curves.append({"teacher": teacher, "seed": seed, "updates": row["updates"],
                                   "decision_nodes_seen": row["decision_nodes_seen"],
                                   "validation_accuracy": row["validation_local"]["accuracy"],
                                   "validation_success": row["validation_rollouts"]["success_rate"],
                                   "validation_loss": row["validation_local"]["loss"]})
    with (Path(root) / "learning-curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for teacher in TEACHERS:
        for ax, metric in zip(axes, ("validation_accuracy", "validation_success")):
            x, center, low, high = [], [], [], []
            for checkpoint in config["checkpoints"]:
                values = [r[metric] for r in curves if r["teacher"] == teacher and r["updates"] == checkpoint]
                x.append(checkpoint * config["batch_decisions"] / 1e6)
                center.append(np.mean(values)); low.append(min(values)); high.append(max(values))
            line, = ax.plot(x, center, marker="o", markersize=3, label=teacher)
            ax.fill_between(x, low, high, color=line.get_color(), alpha=.1)
    for ax in axes:
        ax.set_xscale("log"); ax.set_ylim(-.02, 1.02); ax.grid(alpha=.2)
        ax.set_xlabel("Decisiones supervisadas (millones; escala log)")
    axes[0].set_ylabel("Exactitud de elecciones locales (validación)")
    axes[1].set_ylabel("Ejecuciones correctas (validación)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Media y rango de {len(config['seeds'])} inicializaciones; validación para seguimiento")
    fig.tight_layout()
    fig.savefig(Path(root) / "learning-curves.png", dpi=170)
    fig.savefig(Path(root) / "learning-curves.pdf")
    plt.close(fig)


def run(config, root):
    check_config(config)
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != config:
        raise RuntimeError("Existing comparison has another plan")
    atomic_json(plan_path, config)
    metadata = source_metadata()
    for relative, expected in metadata["source_hashes"].items():
        target = root / "sources" / relative
        if target.exists() and file_hash(target) != expected:
            raise RuntimeError("Archived sources differ")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(Path(relative).read_bytes())
    paths, jobs, excluded, common_graphs = [], [], set(), None
    for teacher in TEACHERS:
        pool, validation = build_pool(config, root, teacher, "train"), build_pool(config, root, teacher, "validation")
        source_fps = {r["fingerprint"] for r in pool.manifest["records"]}
        val_fps = {r["fingerprint"] for r in validation.manifest["records"]}
        if source_fps & val_fps:
            raise RuntimeError("Training/validation overlap")
        ingredients = digest([[r["graph_seed"], [t["tape_seed"] for t in r["trajectories"]]]
                              for r in pool.manifest["records"]])
        if common_graphs is not None and common_graphs != ingredients:
            raise RuntimeError("Teachers do not use paired source graphs/tapes")
        common_graphs = ingredients
        excluded.update(source_fps | val_fps)
        for seed in config["seeds"]:
            jobs.append((config, root, teacher, seed, pool.manifest, validation.manifest, metadata))
    if config["workers"] == 1:
        paths = [train_worker(*job) for job in jobs]
    else:
        # Spawn avoids inheriting PyTorch thread pools or shared RNG state.
        with ProcessPoolExecutor(max_workers=config["workers"],
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(train_worker, *job) for job in jobs]
            paths = [future.result() for future in futures]
        del pool, validation
    excluded.update(fingerprint(g) for g, _ in validation_cases(config))
    frozen = freeze(config, root, paths, metadata)
    summary = summarize(config, root, evaluate(config, root, frozen, excluded))
    verify_runtime(metadata)
    for item in frozen["models"]:
        if file_hash(root / item["path"]) != item["sha256"]:
            raise RuntimeError("Weights changed after freezing")
    print(json.dumps({"status": "complete", "models": len(paths), "cases": summary["cases"],
                      "evaluations": summary["evaluation_rows"], "summary": str(root / "summary.json")}), flush=True)
    return summary
