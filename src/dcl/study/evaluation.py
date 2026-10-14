"""Closed-loop learned execution; exact rules are read-only observers."""
from collections import defaultdict
import time
import numpy as np
import torch

from dcl.counterfactual import exact_commit_labels
from dcl.m4_data import encode_m4_state
from .config import seed_for
from .graphs import generate_graph, fingerprint, degree_stats


def available_matrix(colors, edges, num_colors):
    available = np.ones((len(colors), num_colors), dtype=bool)
    src, dst = edges
    fixed = colors[src] >= 0
    available[dst[fixed], colors[src[fixed]]] = False
    return available


@torch.inference_mode()
def learned_predictions(model, x, edges):
    p, c, commit = model(x, edges, hard_messages=True)
    return ((p > 0).cpu().numpy(), c.argmax(1).cpu().numpy(), (commit > 0).cpu().numpy())


def observe(colors, state, rp, rc, p, c, commit, num_colors):
    uncolored = colors < 0
    available = available_matrix(colors, state.edge_index, num_colors)
    k = available.sum(1)
    chosen_rank = np.floor(rc * k).astype(int)
    ranked = available.cumsum(1) - 1
    truth_offer = ((available & (ranked == chosen_rank[:, None])).argmax(1) + 1)
    truth_offer[(~uncolored) | (k == 0)] = 0
    truth_p = uncolored & (k > 0) & (rp < 0.5)
    valid = exact_commit_labels(state.x, state.edge_index, p, c).astype(bool)
    executed = uncolored & p & (c > 0) & commit
    forbidden = (c > 0) & ~available[np.arange(len(c)), np.maximum(c - 1, 0)]
    src, dst = state.edge_index
    fixed_conflict, simultaneous = np.zeros(len(colors), bool), np.zeros(len(colors), bool)
    np.logical_or.at(fixed_conflict, dst, (colors[src] >= 0) & (colors[src] == c[dst] - 1) & (c[dst] > 0))
    np.logical_or.at(simultaneous, dst, (colors[src] < 0) & p[src] & (c[src] == c[dst]) & (c[dst] > 0))
    by_k = {}
    for value in range(num_colors + 1):
        mask = uncolored & (k == value)
        by_k[str(value)] = {
            "states": int(mask.sum()), "candidate_correct": int((mask & (c == truth_offer)).sum()),
            "forbidden": int((mask & forbidden).sum()), "offers": int((mask & (c > 0)).sum()),
            "executed": int((mask & executed).sum()), "unsafe": int((mask & executed & ~valid).sum()),
            "valid": int((mask & valid).sum()), "unnecessary_wait": int((mask & valid & ~executed).sum()),
        }
    return {
        "states": int(uncolored.sum()), "participation_correct": int((uncolored & (p == truth_p)).sum()),
        "candidate_correct": int((uncolored & (c == truth_offer)).sum()),
        "commit_correct_own": int((uncolored & (commit == valid)).sum()),
        "unsafe_commits": int((executed & ~valid).sum()),
        "fixed_neighbor_conflicts": int((executed & fixed_conflict).sum()),
        "simultaneous_conflicts": int((executed & simultaneous).sum()),
        "valid_commits_rejected": int((valid & ~executed).sum()),
        "by_k": by_k,
    }


def run_rollout(model, graph, tape_seed, num_colors, max_rounds=256, diagnostics=True):
    model.eval()
    device = next(model.parameters()).device
    colors = np.full(len(graph), -1, np.int64)
    rng = np.random.default_rng(tape_seed)
    totals, by_k = defaultdict(int), defaultdict(lambda: defaultdict(int))
    inference_seconds = 0.0
    started = time.perf_counter()
    for round_i in range(1, max_rounds + 1):
        tick = time.perf_counter()
        # Two independent draws per node per round, with no oracle invocation.
        rp, rc = rng.random(len(graph)), rng.random(len(graph))
        state = encode_m4_state(graph, colors, rp, rc, num_colors)
        x = torch.from_numpy(state.x).to(device)
        edges = torch.from_numpy(state.edge_index).to(device)
        p, c, commit = learned_predictions(model, x, edges)
        executed = (colors < 0) & p & (c > 0) & commit
        # The action is fully determined before the read-only diagnostic.
        next_colors = colors.copy()
        next_colors[executed] = c[executed] - 1
        inference_seconds += time.perf_counter() - tick
        if diagnostics:
            observed = observe(colors, state, rp, rc, p, c, commit, num_colors)
            for key, value in observed.items():
                if key != "by_k":
                    totals[key] += value
            for value, counts in observed["by_k"].items():
                for key, count in counts.items():
                    by_k[value][key] += count
        colors = next_colors
        if np.all(colors >= 0):
            break
    bad_edges = sum(colors[u] >= 0 and colors[u] == colors[v] for u, v in graph.edges())
    finished = bool(np.all(colors >= 0))
    return {"finished": finished, "success": finished and bad_edges == 0,
            "monochromatic_edges": int(bad_edges), "rounds": round_i,
            "uncolored_nodes": int((colors < 0).sum()), "inference_seconds": inference_seconds,
            "wall_seconds": time.perf_counter() - started, **dict(totals),
            "by_k": {k: dict(v) for k, v in by_k.items()}, "colors": colors}


def evaluation_cases(config, split, sizes, counts, test_seed_namespace=20261014):
    if split not in ("validation_rollout", "final_test"):
        raise ValueError("Unknown evaluation split")
    # Final graphs/tapes are identical for all matched models, including
    # all seeds. Validation is replica-specific and independent of training.
    replica = config.seed if split == "validation_rollout" else test_seed_namespace
    for n in sizes:
        for family in ("REGULAR", "IRREGULAR"):
            for i in range(counts[n]):
                graph_seed = seed_for(replica, config.degree, f"{split}/graph", n, family, i)
                tape_seed = seed_for(replica, config.degree, f"{split}/tape", n, family, i)
                graph = generate_graph(n, config.degree, family, graph_seed)
                yield graph, {"graph_id": f"{split}/d{config.degree}/{family}/n{n}/{i}",
                              "graph_seed": graph_seed, "tape_seed": tape_seed, "n": n,
                              "degree": config.degree, "distribution": family,
                              "fingerprint": fingerprint(graph), **degree_stats(graph)}


def validate_rollouts(model, config):
    rows = []
    for graph, metadata in evaluation_cases(config, "validation_rollout", config.val_sizes,
                                            dict.fromkeys(config.val_sizes, config.val_rollouts)):
        result = run_rollout(model, graph, metadata["tape_seed"], config.num_colors, config.max_rounds)
        result.pop("colors")
        rows.append({**metadata, **result})
    successes = sum(row["success"] for row in rows)
    states = sum(row["states"] for row in rows)
    return {"successes": successes, "graphs": len(rows), "success_rate": successes / len(rows),
            "unsafe_commits": sum(row["unsafe_commits"] for row in rows),
            "unsafe_per_state": sum(row["unsafe_commits"] for row in rows) / max(1, states),
            "mean_rounds": float(np.mean([r["rounds"] for r in rows])), "per_graph": rows}
