"""Compare the five original frozen E1 networks with Luby, without training.

E1 means UniversalColorGNN NORMALIZED / COUNTERFACTUAL, seeds 0..4, epoch
120. Only checkpoints authenticated by Martín's E1_INDEX.json are accepted.
The model's own actions evolve its state; exact rules only observe them.
"""
from collections import defaultdict
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
import torch

from dcl.m4_data import _edge_index  # Graph serialization only; no M4B model.
from dcl.study.config import digest, seed_for
from dcl.study.graphs import degree_stats, fingerprint, generate_graph
from dcl.study.storage import atomic_json, atomic_write, file_hash, now
from dcl.universal.config import UConfig
from dcl.universal.model import UniversalColorGNN, count_parameters
from dcl.universal.rules import available_matrix, exact_commit_labels, teacher_offer
from dcl.universal.train import load_model
from dcl.behavior_statistics import canonical_colors, coupling_bound, empirical_tv, holm_adjust, kernel_test, write_csv

PROTOCOL = "e1-luby-behavior-v1"
UPSTREAM = "1d8857b25e8b654e87351fce33813692c70a6250"
REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "results/ucgnn/E1_INDEX.json"
SOURCE_RECORD = REPO / "results/e1-behavior/2026-10-21/upstream-source.json"
SUCCESS, COLLISION, TIMEOUT = 0, 1, 2
LOCAL_METRICS = ("states", "participation_errors", "candidate_errors", "action_errors",
                 "teacher_participants", "shared_commit_errors", "own_attempts",
                 "unsafe_commits", "missed_safe_commits", "forbidden_offers")


def named_seed(plan, *coordinates):
    return int(digest([PROTOCOL, plan["namespace"], *coordinates])[:16], 16)


def validate_plan(plan):
    if plan["protocol"] != PROTOCOL or (plan["degree"], plan["num_colors"]) != (3, 4):
        raise ValueError("This E1 comparison fixes degree 3 and four colors")
    if plan["replicas"] != list(range(5)):
        raise ValueError("All five original E1 seeds 0..4 are prespecified")
    sizes = [n for n, _ in plan["graphs_per_family"]]
    if len(set(sizes)) != len(sizes) or any(n < 6 or n % 2 or c < 1 for n, c in plan["graphs_per_family"]):
        raise ValueError("Distinct valid graph sizes/counts required")
    for key in ("samples", "max_rounds", "batch_nodes", "batch_tapes", "permutations"):
        if plan[key] < 1:
            raise ValueError(f"{key} must be positive")
    if plan["samples"] < 2 or not 0 < plan["alpha"] < 1 or not 0 < plan["tv_tolerance"] < 1:
        raise ValueError("Invalid sampling or inference settings")
    if not plan["kernel_scales"] or min(plan["kernel_scales"]) <= 0:
        raise ValueError("Positive kernel scales required")


def inspect_inputs(weights_root=REPO, replication_index=None):
    """Read-only preflight; never initialize substitute experimental weights."""
    index = json.loads(INDEX.read_text())
    record = json.loads(SOURCE_RECORD.read_text())
    if record["commit"] != UPSTREAM:
        raise RuntimeError("Wrong upstream source commit")
    for path, expected in record["files"].items():
        if file_hash(REPO / path) != expected:
            raise RuntimeError(f"Original E1 source changed: {path}")
    for path in ("src/dcl/universal/model.py", "src/dcl/universal/train.py", "configs/ucgnn/e1.json",
                 "configs/ucgnn/replication.json"):
        if file_hash(REPO / path) != index["sources"][path]:
            raise RuntimeError(f"E1_INDEX source mismatch: {path}")
    if (index["experiment"], index["architecture"], index["method"], index["parameters"],
        index["checkpoint_epoch"], index["networks"]) != (
            "E1", "UniversalColorGNN NORMALIZED", "COUNTERFACTUAL", 16035, 120, 5):
        raise RuntimeError("Wrong E1 identity")
    if [x["seed"] for x in index["checkpoints"]] != list(range(5)):
        raise RuntimeError("Wrong original E1 seed inventory")
    config0 = UConfig.from_dict(json.loads((REPO / "configs/ucgnn/e1.json").read_text())["experiments"][0])
    origin, index_path = "historical-original", INDEX
    if replication_index is not None:
        index_path = Path(replication_index)
        replicated = json.loads(index_path.read_text())
        if (replicated.get("protocol"), replicated.get("cohort_origin"), replicated.get("upstream_commit")) != (
                "e1-replication-v1", "newly-trained-replicas", UPSTREAM):
            raise RuntimeError("Invalid new E1 cohort identity")
        if replicated["canonical_config_sha256"] != file_hash(REPO / "configs/ucgnn/e1.json"):
            raise RuntimeError("Replicated E1 configuration differs from Martín's")
        for path in ("src/dcl/universal/model.py", "src/dcl/universal/train.py", "src/dcl/universal/config.py",
                     "src/dcl/universal/data.py", "src/dcl/universal/rules.py"):
            if replicated["sources"][path] != file_hash(REPO / path):
                raise RuntimeError(f"Replicated E1 source changed: {path}")
        if not replicated["frozen_before_comparison"] or [x["seed"] for x in replicated["checkpoints"]] != list(range(5)):
            raise RuntimeError("Five final E1 replicas must be frozen before the test")
        index = replicated
        origin = "newly-trained-replicas"
    checkpoints = []
    for item in index["checkpoints"]:
        config = replace(config0, seed=item["seed"])
        expected_path = (("data", "ucgnn", "runs", config.id, "epoch-120.pt") if origin == "historical-original"
                         else ("runs", config.id, "epoch-120.pt"))
        if config.id != item["run"] or Path(item["checkpoint"]).parts != expected_path:
            raise RuntimeError("Wrong E1 run/path in index")
        path = Path(weights_root) / item["checkpoint"]
        state = "missing"
        if path.exists():
            state = "verified" if path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"] else "mismatch"
        checkpoints.append({**item, "resolved_path": str(path.resolve()), "state": state})
    return {"experiment": "E1", "architecture": index["architecture"], "method": index["method"],
            "parameters": index["parameters"], "epoch": 120, "upstream_commit": UPSTREAM,
            "source_verified": True, "index_sha256": file_hash(index_path), "cohort_origin": origin,
            "cohort_index": str(index_path.resolve()), "checkpoints": checkpoints,
            "ready": all(item["state"] == "verified" for item in checkpoints)}


def load_models(weights_root=REPO, replication_index=None):
    inputs = inspect_inputs(weights_root, replication_index)
    if not inputs["ready"]:
        missing = ", ".join(f"s{x['seed']}={x['state']}" for x in inputs["checkpoints"] if x["state"] != "verified")
        raise RuntimeError(f"Original E1 weights unavailable ({missing}); no comparison was run")
    models = {}
    for item in inputs["checkpoints"]:
        model, saved = load_model(item["resolved_path"], "cpu")
        config = UConfig.from_dict(saved["config"])
        if (config.id, saved["epoch"], saved["parameters"], saved["complete"], count_parameters(model)) != (
                item["run"], 120, 16035, True, 16035):
            raise RuntimeError("Authenticated checkpoint metadata does not describe the original final E1")
        if type(model) is not UniversalColorGNN or model.variant != "NORMALIZED":
            raise RuntimeError("Wrong E1 architecture")
        models[item["seed"]] = model.eval()
    return models, inputs


def batch_edges(edges, batch, n):
    return np.ascontiguousarray((edges[None] + np.arange(batch)[:, None, None] * n).transpose(1, 0, 2).reshape(2, -1))


def truth_from_draws(colors, rp, rc, edges, palette=4):
    """Exact E1 teacher rules with original float64 random draws."""
    flat = colors.ravel()
    repeated = batch_edges(edges, len(colors), colors.shape[1])
    c, k = teacher_offer(flat, rc.ravel(), repeated, palette)
    p = (flat < 0) & (k > 0) & (rp.ravel() < 0.5)
    commit = exact_commit_labels(flat, repeated, p, c)
    available = available_matrix(flat, repeated, palette)
    return repeated, p.reshape(colors.shape), c.reshape(colors.shape), commit.reshape(colors.shape), k.reshape(colors.shape), available


def bad_edges(colors, edges):
    src, dst = edges[:, edges[0] < edges[1]]
    return ((colors[:, src] >= 0) & (colors[:, src] == colors[:, dst])).sum(1)


def step_colors(colors, p, c, commit, alive):
    result = colors.copy()
    execute = (colors < 0) & p & (c >= 0) & commit & alive[:, None]
    result[execute] = c[execute]
    return result


def mark_finished(colors, alive, rounds, r):
    # Match the original E1 run_rollout: continue after a collision until all
    # vertices have fixed a color or the round cap is reached.
    finished = alive & (colors >= 0).all(1)
    rounds[finished] = r
    alive[finished] = False


def terminal_status(colors, edges):
    return np.where(bad_edges(colors, edges) > 0, COLLISION,
                    np.where((colors >= 0).all(1), SUCCESS, TIMEOUT)).astype(np.int8)


def draws(generators, n):
    array = np.stack([np.stack((rng.random(n), rng.random(n))) for rng in generators])
    return array[:, 0], array[:, 1]


def teacher_batch(graph, seeds, plan):
    colors = np.full((len(seeds), len(graph)), -1, np.int64)
    alive = np.ones(len(seeds), bool)
    rounds = np.full(len(seeds), plan["max_rounds"], np.int16)
    generators, edges = [np.random.default_rng(int(s)) for s in seeds], _edge_index(graph)
    for r in range(1, plan["max_rounds"] + 1):
        rp, rc = draws(generators, len(graph))
        _, p, c, commit, _, _ = truth_from_draws(colors, rp, rc, edges, plan["num_colors"])
        colors = step_colors(colors, p, c, commit, alive)
        mark_finished(colors, alive, rounds, r)
        if not alive.any():
            break
    status = terminal_status(colors, edges)
    if np.any(status == COLLISION):
        raise RuntimeError("Exact Luby produced a conflict")
    return {"colors": colors, "rounds": rounds, "status": status}


def count_local(counts, examples, replica, source, colors, alive, rp, rc,
                truth_p, truth_c, truth_commit, k, available, p, c, commit,
                shared_commit, valid_own, r, graph_id):
    active = (colors < 0) & alive[:, None]
    execute = active & p & (c >= 0) & commit
    forbidden = (c >= 0) & ~available[np.arange(c.size), np.maximum(c.ravel(), 0)].reshape(c.shape)
    masks = {
        "states": active, "participation_errors": active & (p != truth_p),
        "candidate_errors": active & (c != truth_c),
        "action_errors": active & (np.where(execute, c, -1) != np.where(truth_commit, truth_c, -1)),
        "teacher_participants": active & truth_p,
        "shared_commit_errors": active & truth_p & (shared_commit != truth_commit),
        "own_attempts": active & p & (c >= 0), "unsafe_commits": execute & ~valid_own,
        "missed_safe_commits": active & valid_own & ~execute, "forbidden_offers": active & forbidden,
    }
    for value in range(-1, available.shape[1] + 1):
        stratum = np.ones_like(active) if value == -1 else k == value
        bucket = counts[(replica, source, value)]
        for name, mask in masks.items():
            bucket[name] += int((mask & stratum).sum())
    for metric in ("participation_errors", "candidate_errors", "shared_commit_errors", "unsafe_commits"):
        key = (replica, source, metric)
        if key not in examples and masks[metric].any():
            b, v = np.argwhere(masks[metric])[0]
            examples[key] = {"replica": replica, "source": source, "metric": metric, "graph_id": graph_id,
                             "round": r, "batch_index": int(b), "vertex": int(v),
                             "colors_before": colors[b].tolist(), "activation_draw": rp[b].tolist(),
                             "candidate_draw": rc[b].tolist(), "truth_p": truth_p[b].tolist(),
                             "truth_c": truth_c[b].tolist(), "truth_commit": truth_commit[b].tolist(),
                             "pred_p": p[b].tolist(), "pred_c": c[b].tolist(), "pred_commit": commit[b].tolist(),
                             "shared_commit": shared_commit[b].tolist()}


@torch.inference_mode()
def coupled_batch(graph, seeds, models, plan, graph_id="test"):
    n, batch = len(graph), len(seeds)
    teacher_colors = np.full((batch, n), -1, np.int64)
    teacher_alive, teacher_rounds = np.ones(batch, bool), np.full(batch, plan["max_rounds"], np.int16)
    learned = {replica: {"colors": teacher_colors.copy(), "alive": teacher_alive.copy(),
                        "rounds": teacher_rounds.copy(), "first_action_round": np.zeros(batch, np.int16),
                        "first_decision_round": np.zeros(batch, np.int16)} for replica in models}
    counts, examples = defaultdict(lambda: defaultdict(int)), {}
    generators = [np.random.default_rng(int(seed)) for seed in seeds]
    edges = _edge_index(graph)
    repeated_t = torch.from_numpy(batch_edges(edges, 2 * batch, n))
    palette_t = torch.full((2 * batch * n,), plan["num_colors"], dtype=torch.long)
    for r in range(1, plan["max_rounds"] + 1):
        rp, rc = draws(generators, n)
        _, tp, tc, tm, _, _ = truth_from_draws(teacher_colors, rp, rc, edges, plan["num_colors"])
        for replica, model in models.items():
            state = learned[replica]
            colors = np.concatenate((teacher_colors, state["colors"]))
            alive = np.concatenate((teacher_alive, state["alive"]))
            rp2, rc2 = np.concatenate((rp, rp)), np.concatenate((rc, rc))
            e, truth_p, truth_c, truth_commit, k, available = truth_from_draws(colors, rp2, rc2, edges, plan["num_colors"])
            colors_t = torch.from_numpy(colors.ravel())
            # Exact original E1 input conversion: float64 tape -> float32 model.
            out = model(colors_t, torch.from_numpy(rp2.ravel().astype(np.float32)),
                        torch.from_numpy(rc2.ravel().astype(np.float32)), repeated_t, palette_t)
            shape = colors.shape
            p, c = out.participate.numpy().reshape(shape), out.candidate.numpy().reshape(shape)
            commit = (out.commit_logit > 0).numpy().reshape(shape)
            shared_commit = (model.commit_logits(colors_t, repeated_t, palette_t,
                              torch.from_numpy(truth_p.ravel()), torch.from_numpy(truth_c.ravel())) > 0).numpy().reshape(shape)
            valid_own = exact_commit_labels(colors.ravel(), e, p.ravel(), c.ravel()).reshape(shape)
            for source, section in (("teacher_states", slice(0, batch)), ("gnn_states", slice(batch, 2 * batch))):
                av = available.reshape(2 * batch, n, plan["num_colors"])[section].reshape(-1, plan["num_colors"])
                count_local(counts, examples, replica, source, colors[section], alive[section], rp2[section], rc2[section],
                            truth_p[section], truth_c[section], truth_commit[section], k[section], av,
                            p[section], c[section], commit[section], shared_commit[section], valid_own[section], r, graph_id)
            gp, gc, gm = p[batch:], c[batch:], commit[batch:]
            active = (teacher_colors < 0) & teacher_alive[:, None]
            ta = np.where(tm, tc, -1)
            decision_diff = (active & ((p[:batch] != tp) | (c[:batch] != tc) |
                             (np.where(p[:batch] & commit[:batch] & (c[:batch] >= 0), c[:batch], -1) != ta))).any(1)
            state["first_decision_round"][(state["first_decision_round"] == 0) & decision_diff] = r
            action_diff = (active & (np.where(gp & gm & (gc >= 0), gc, -1) != ta)).any(1)
            first = (state["first_action_round"] == 0) & (state["colors"] == teacher_colors).all(1) & state["alive"] & teacher_alive & action_diff
            state["first_action_round"][first] = r
            state["colors"] = step_colors(state["colors"], gp, gc, gm, state["alive"])
            mark_finished(state["colors"], state["alive"], state["rounds"], r)
        teacher_colors = step_colors(teacher_colors, tp, tc, tm, teacher_alive)
        mark_finished(teacher_colors, teacher_alive, teacher_rounds, r)
        if not teacher_alive.any() and not any(state["alive"].any() for state in learned.values()):
            break
    for state in learned.values():
        state.pop("alive")
        state["status"] = terminal_status(state["colors"], edges)
    teacher = {"colors": teacher_colors, "rounds": teacher_rounds, "status": terminal_status(teacher_colors, edges)}
    if np.any(teacher["status"] == COLLISION):
        raise RuntimeError("Exact reference produced a conflict")
    return teacher, learned, counts, examples


def build_graphs(plan):
    """Exclude all original training/validation fingerprints, without training."""
    config0 = UConfig.from_dict(json.loads((REPO / "configs/ucgnn/e1.json").read_text())["experiments"][0])
    excluded = set()
    for replica in plan["replicas"]:
        config = replace(config0, seed=replica)
        for split, count in (("train", config.graphs_per_degree[0]), ("validation", config.val_graphs_per_degree)):
            for i in range(count):
                family = config.family if split == "train" else ("REGULAR" if i % 2 == 0 else "IRREGULAR")
                graph = generate_graph(config.n, 3, family, seed_for(replica, 3, f"{split}/graph", i))
                excluded.add(fingerprint(graph))
    graphs, seen = [], set()
    for n, count in plan["graphs_per_family"]:
        for family in ("REGULAR", "IRREGULAR"):
            for i in range(count):
                attempt = 0
                while True:
                    seed = named_seed(plan, "graph", n, family, i, attempt)
                    graph = generate_graph(n, 3, family, seed)
                    fp = fingerprint(graph)
                    if fp not in excluded and fp not in seen:
                        break
                    attempt += 1
                seen.add(fp)
                graphs.append((graph, {"id": f"{family}-n{n}-g{i}", "family": family, "n": n,
                                       "graph_i": i, "graph_seed": seed, "fingerprint": fp,
                                       "role": "random", "attempt": attempt, **degree_stats(graph)}))
    if plan["controls"]:
        for family, graph in (("PATH", nx.path_graph(8)), ("CYCLE", nx.cycle_graph(8)), ("CLIQUE", nx.complete_graph(4))):
            graphs.append((graph, {"id": f"{family}-n{len(graph)}-g0", "family": family, "n": len(graph),
                                   "graph_i": 0, "graph_seed": None, "fingerprint": fingerprint(graph),
                                   "role": "control", **degree_stats(graph)}))
    return graphs, {"count": len(excluded), "sha256_digest": digest(sorted(excluded)),
                    "definition": "all 800 train + 200 validation graphs per original seed, labeled graph fingerprints"}


def source_hashes():
    upstream_paths = json.loads(SOURCE_RECORD.read_text())["files"]
    paths = [*upstream_paths, "src/dcl/e1_behavior.py", "src/dcl/behavior_statistics.py",
             "experiments/compare_e1_behavior.py", "configs/e1_behavior/comparison.json", "tests/test_e1_behavior.py",
             "src/dcl/m4_data.py", "src/dcl/oracle.py", "src/dcl/study/config.py", "src/dcl/study/storage.py",
             "src/dcl/study/graphs.py", "src/dcl/graphs.py", str(SOURCE_RECORD.relative_to(REPO))]
    return {p: file_hash(REPO / p) for p in paths}


def prepare(plan, root, weights_root=REPO, replication_index=None):
    validate_plan(plan)
    models, inputs = load_models(weights_root, replication_index)
    graphs, exclusion = build_graphs(plan)
    manifest = {"protocol": PROTOCOL, "plan": plan, "inputs": inputs, "models": inputs["checkpoints"],
                "sources": source_hashes(), "versions": {n: version(n) for n in ("torch", "numpy", "scipy", "networkx")},
                "exclusion": exclusion, "graphs": [meta for _, meta in graphs],
                "selection": f"all five final E1 networks ({inputs.get('cohort_origin', 'historical-original')}), no model or test selection",
                "independence": "Luby A and E1 B have independent tapes; Luby B supplies the null and same-tape comparison",
                "termination": "original E1 semantics: continue after collisions until finished or capped"}
    path = Path(root) / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("Existing manifest differs in plan, model, sources, runtime or graphs")
    else:
        atomic_json(path, manifest)
    for source, expected in manifest["sources"].items():
        destination = Path(root) / "sources" / source
        if destination.exists():
            if file_hash(destination) != expected:
                raise RuntimeError("Source snapshot changed")
        else:
            atomic_write(destination, lambda f, p=source: f.write((REPO / p).read_bytes()))
    return models, graphs, manifest


def run(plan_path, root, weights_root=REPO, replication_index=None):
    plan, root = json.loads(Path(plan_path).read_text()), Path(root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models, graphs, manifest = prepare(plan, root, weights_root, replication_index)
    started = time.perf_counter()
    for graph_i, (graph, meta) in enumerate(graphs, 1):
        path = root / "graphs" / (meta["id"] + ".json")
        arrays_path = path.with_suffix(".npz")
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["manifest_hash"] != digest(manifest) or previous["arrays_sha256"] != file_hash(arrays_path):
                raise RuntimeError("Existing graph result changed")
            continue
        seeds_a = np.array([named_seed(plan, "tape-A", meta["id"], j) for j in range(plan["samples"])], np.uint64)
        seeds_b = np.array([named_seed(plan, "tape-B", meta["id"], j) for j in range(plan["samples"])], np.uint64)
        batch_size = max(1, min(plan["batch_tapes"], plan["batch_nodes"] // len(graph)))
        arrays, counts, examples = defaultdict(list), defaultdict(lambda: defaultdict(int)), {}
        for start in range(0, plan["samples"], batch_size):
            stop = start + batch_size
            a = teacher_batch(graph, seeds_a[start:stop], plan)
            b, learned, local, sample_examples = coupled_batch(graph, seeds_b[start:stop], models, plan, meta["id"])
            for actor, result in [("luby_a", a), ("luby_b", b), *((f"gnn_s{s}", learned[s]) for s in models)]:
                for key, value in result.items():
                    arrays[f"{actor}_{key}"].append(value)
            for key, values in local.items():
                for metric, value in values.items():
                    counts[key][metric] += value
            for key, example in sample_examples.items():
                example["tape_i"] = start + example.pop("batch_index")
                examples.setdefault(key, example)
        arrays = {key: np.concatenate(value) for key, value in arrays.items()}
        arrays.update(tape_seeds_a=seeds_a, tape_seeds_b=seeds_b, edges=_edge_index(graph))
        atomic_write(arrays_path, lambda f: np.savez_compressed(f, **arrays))
        atomic_json(path, {"case": meta, "manifest_hash": digest(manifest), "arrays_sha256": file_hash(arrays_path),
                          "local": [{"replica": s, "source": source, "k": k, **dict(values)}
                                    for (s, source, k), values in sorted(counts.items())],
                          "examples": list(examples.values())})
        print(f"E1 graph {graph_i}/{len(graphs)} {meta['id']} complete; elapsed={time.perf_counter() - started:.1f}s", flush=True)
    if not inspect_inputs(weights_root, replication_index)["ready"]:
        raise RuntimeError("A frozen E1 checkpoint changed during evaluation")
    if source_hashes() != manifest["sources"]:
        raise RuntimeError("An evaluation source changed during evaluation")
    atomic_json(root / "execution.json", {"complete": True, "graphs": len(graphs), "created_utc": now(),
                "manifest_hash": digest(manifest), "wall_seconds": time.perf_counter() - started,
                "rollouts": len(graphs) * plan["samples"] * (2 + len(models))})


def summarize(root, output):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    execution = json.loads((root / "execution.json").read_text())
    if not execution["complete"] or execution["manifest_hash"] != digest(manifest):
        raise RuntimeError("Complete authenticated E1 execution required")
    plan = manifest["plan"]
    validate_plan(plan)
    torch.set_num_threads(1)
    graph_rows, local_rows, examples, strata = [], [], {}, defaultdict(list)
    total_bounds = len(manifest["graphs"]) * len(plan["replicas"]) * 2
    for i, meta in enumerate(manifest["graphs"], 1):
        path = root / "graphs" / (meta["id"] + ".json")
        result = json.loads(path.read_text())
        if result["manifest_hash"] != digest(manifest) or file_hash(path.with_suffix(".npz")) != result["arrays_sha256"]:
            raise RuntimeError("Corrupt E1 graph result")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as stored:
            arrays = dict(stored)
        for actor in ("luby_a", "luby_b", *(f"gnn_s{s}" for s in plan["replicas"])):
            if not np.array_equal(terminal_status(arrays[actor + "_colors"], arrays["edges"]), arrays[actor + "_status"]):
                raise RuntimeError("Status disagrees with the complete terminal coloring")
        local_rows.extend({**meta, **row} for row in result["local"])
        for example in result["examples"]:
            examples.setdefault((example["replica"], example["source"], example["metric"]), example)
        a, reference = arrays["luby_a_colors"], arrays["luby_b_colors"]
        for replica in [None, *plan["replicas"]]:
            actor = "luby_b" if replica is None else f"gnn_s{replica}"
            b = arrays[actor + "_colors"]
            sa, sb = arrays["luby_a_status"], arrays[actor + "_status"]
            for representation in ("labelled", "color_permutation"):
                aa, bb, rr = ((a, b, reference) if representation == "labelled" else
                              (canonical_colors(a), canonical_colors(b), canonical_colors(reference)))
                stat = kernel_test(aa, bb, sa, sb, plan, np.random.default_rng(
                    named_seed(plan, "permutation", meta["id"], replica, representation)))
                row = {**meta, "replica": replica, "actor": actor, "representation": representation,
                       "samples": len(a), "mmd2": stat["mmd2"], "mmd_null95": stat["null95"], "p_value": stat["p_value"],
                       "empirical_tv": empirical_tv(np.column_stack((aa, sa)), np.column_stack((bb, sb))),
                       "success_rate": float(np.mean(sb == SUCCESS)), "collision_rate": float(np.mean(sb == COLLISION)),
                       "timeout_rate": float(np.mean(sb == TIMEOUT)), "mean_rounds_all": float(arrays[actor + "_rounds"].mean()),
                       "luby_mean_rounds": float(arrays["luby_a_rounds"].mean())}
                if replica is not None:
                    errors = int(((bb != rr).any(1) | (sb != arrays["luby_b_status"])).sum())
                    upper = coupling_bound(errors, len(a), plan["alpha"] / total_bounds)
                    row.update(coupled_mismatches=errors, coupled_mismatch_rate=errors / len(a),
                               coupling_upper95=coupling_bound(errors, len(a), plan["alpha"]),
                               coupling_upper_simultaneous95=upper,
                               tv_within_tolerance_certified=upper <= plan["tv_tolerance"],
                               first_action_divergences=int((arrays[actor + "_first_action_round"] > 0).sum()),
                               first_decision_divergences=int((arrays[actor + "_first_decision_round"] > 0).sum()))
                graph_rows.append(row)
                strata[(meta["role"], meta["family"], meta["n"], replica, representation)].append(stat)
        if i % 5 == 0 or i == len(manifest["graphs"]):
            print(f"E1 distribution tests {i}/{len(manifest['graphs'])} graphs", flush=True)
    test_rows = []
    for (role, family, n, replica, representation), values in strata.items():
        observed, null = float(np.mean([v["mmd2"] for v in values])), np.mean([v["null"] for v in values], axis=0)
        test_rows.append({"role": role, "family": family, "n": n, "replica": replica, "representation": representation,
                          "graphs": len(values), "mmd2": observed, "null95": float(np.quantile(null, 0.95)),
                          "p_value": float((1 + np.count_nonzero(null >= observed - 1e-14)) / (len(null) + 1))})
    primary = [r for r in test_rows if r["role"] == "random" and r["replica"] is not None]
    for row, p in zip(primary, holm_adjust([r["p_value"] for r in primary])):
        row.update(p_holm=float(p), different_after_holm=bool(p <= plan["alpha"]))
    aggregate_local = defaultdict(lambda: defaultdict(int))
    for row in local_rows:
        if row["role"] == "random":
            for metric in LOCAL_METRICS:
                aggregate_local[(row["replica"], row["source"], row["k"])][metric] += row[metric]
    local_summary = []
    for (replica, source, k), counts in sorted(aggregate_local.items()):
        row = {"replica": replica, "source": source, "k": k, **dict(counts)}
        for metric in ("participation", "candidate", "action"):
            row[metric + "_agreement"] = 1 - counts[metric + "_errors"] / counts["states"] if counts["states"] else None
        row["shared_commit_agreement"] = 1 - counts["shared_commit_errors"] / counts["teacher_participants"] if counts["teacher_participants"] else None
        local_summary.append(row)
    origin = manifest["inputs"].get("cohort_origin", "historical-original")
    summary = {"protocol": PROTOCOL, "experiment": "E1", "cohort_origin": origin, "plan": plan, "manifest_hash": digest(manifest),
               "graphs": len(manifest["graphs"]), "rollouts": execution["rollouts"], "primary_tests": len(primary),
               "different_after_holm": sum(r["different_after_holm"] for r in primary),
               "random_graph_equivalence_certificates": sum(r.get("tv_within_tolerance_certified", False)
                                                           for r in graph_rows if r["role"] == "random"),
               "review": "finite experimental evidence; independent scientific review pending"}
    for name, rows in (("by-graph.csv", graph_rows), ("distribution-tests.csv", test_rows),
                       ("local-by-graph.csv", local_rows), ("local-summary.csv", local_summary)):
        write_csv(output / name, rows)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "counterexamples.json", list(examples.values()))
    description = "cinco réplicas E1 nuevas a época120" if origin == "newly-trained-replicas" else "cinco E1 originales a época120"
    report = ["# Comparación E1/Luby", "", f"Evidencia experimental finita, {description}.", "",
              f"{summary['rollouts']} ejecuciones completas; {len(primary)} tests principales; "
              f"{summary['different_after_holm']} diferencias tras Holm.", "",
              "MMD compara muestras independientes dentro de cada grafo. No rechazar igualdad no certifica cercanía.",
              "La TV empírica es descriptiva. Las cotas binomiales del desacuerdo con tapes comunes solo dan una cota "
              "superior de TV, simultánea por Bonferroni; desacuerdo alto no prueba separación distribucional.", "",
              "Las comparaciones locales usan estados y sorteos idénticos, separando trayectorias Luby y propias. "
              "COMMIT se consulta adicionalmente con propuestas exactas compartidas. Los conteos locales están correlacionados.", "",
              "El modelo ejecuta sus propias acciones sin filtro, con la terminación original E1. "
              "Se conservan coloraciones parciales, colisiones y timeouts. Revisión independiente pendiente."]
    atomic_write(output / "REPORT.md", lambda f: f.write(("\n".join(report) + "\n").encode()))
