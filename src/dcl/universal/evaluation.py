"""Closed-loop execution of a learned universal policy; exact rules only observe.

A rollout draws exactly two uniforms per node per round from the tape (the
same consumption as the oracle), lets the model decide PARTICIPATE, CANDIDATE
and COMMIT, and applies the committed colors. No shield, oracle call or mask
other than the declared palette intervenes. Diagnostics are computed from the
exact rules *after* the action is fixed and never alter the state.
"""
from __future__ import annotations

from collections import defaultdict
import time

import numpy as np
import torch

from dcl.m4_data import _edge_index
from dcl.study.config import seed_for
from dcl.study.graphs import degree_stats, fingerprint, generate_graph
from .rules import available_matrix, conflict_flags, exact_commit_labels, teacher_offer

BY_K_KEYS = ("states", "participation_correct", "candidate_correct", "forbidden", "offers", "executed",
             "unsafe", "valid", "unnecessary_wait", "commit_correct", "action_agree", "fixed_conflict_executed",
             "simultaneous_conflict_executed", "availability_slot_errors", "availability_slots",
             "interval_deviation_sum", "interval_deviation_nodes")

# A node's learned partition of [0,1) deviates if some endpoint is off by more than this.
INTERVAL_TOLERANCE = 0.02


def auto_edge_chunk(num_edges, palette, message_dim=32, budget_bytes=192 * 2**20):
    per_edge = palette * message_dim * 4
    return None if num_edges * per_edge <= budget_bytes else max(1, budget_bytes // per_edge)


def teacher_rollout(n, edge_index, palette, tape_seed, max_rounds=1000):
    """Vectorized oracle with the oracle's exact tape consumption (tested equal to run_oracle)."""
    colors = np.full(n, -1, np.int64)
    rng = np.random.default_rng(tape_seed)
    for round_i in range(1, max_rounds + 1):
        rp, rc = rng.random(n), rng.random(n)
        offer, k = teacher_offer(colors, rc, edge_index, palette)
        part = (colors < 0) & (k > 0) & (rp < 0.5)
        commit = exact_commit_labels(colors, edge_index, part, offer)
        colors = colors.copy()
        colors[commit] = offer[commit]
        if np.all(colors >= 0):
            return colors, round_i
    return colors, max_rounds


def observe(colors, edge_index, palette, rp, rc, p, c, commit, alpha, lo, hi):
    """Read-only diagnostics of one learned round, against the exact rules.

    alpha only matters up to a per-node scale (lo/hi are normalized by k_hat),
    so learned availability is alpha >= 0.5 * max_c alpha. The interval
    deviation compares the learned [lo, hi) with the teacher's exact
    [P_<(c)/k, P_<=(c)/k) over the palette.
    """
    n = len(colors)
    uncolored = colors < 0
    available = available_matrix(colors, edge_index, palette)
    offer_t, k = teacher_offer(colors, rc, edge_index, palette)
    part_t = uncolored & (k > 0) & (rp < 0.5)
    action_t = np.where(exact_commit_labels(colors, edge_index, part_t, offer_t), offer_t, -1)
    valid = exact_commit_labels(colors, edge_index, p, c)
    executed = uncolored & p & (c >= 0) & commit
    action = np.where(executed, c, -1)
    fixed, simultaneous = conflict_flags(colors, edge_index, p, c)
    forbidden = (c >= 0) & ~available[np.arange(n), np.maximum(c, 0)]
    relative = alpha / np.maximum(alpha.max(1, keepdims=True), 1e-12)
    slot_errors = ((relative >= 0.5) != available) & uncolored[:, None]
    prefix = available.cumsum(1) - available
    k_safe = np.maximum(k, 1)[:, None]
    deviation = np.maximum(np.abs(lo - prefix / k_safe), np.abs(hi - (prefix + available) / k_safe)).max(1)
    deviation = np.where(uncolored, deviation, 0.0)
    flags = {
        "states": uncolored, "participation_correct": uncolored & (p == part_t),
        "candidate_correct": uncolored & (c == offer_t), "forbidden": uncolored & forbidden,
        "offers": uncolored & (c >= 0), "executed": executed, "unsafe": executed & ~valid, "valid": valid,
        "unnecessary_wait": valid & ~executed, "commit_correct": uncolored & (commit == valid),
        "action_agree": uncolored & (action == action_t), "fixed_conflict_executed": executed & fixed,
        "simultaneous_conflict_executed": executed & simultaneous,
    }
    by_k = {}
    for value in np.unique(k[uncolored]):
        mask = uncolored & (k == value)
        row = {key: int((flag & mask).sum()) for key, flag in flags.items()}
        row["availability_slot_errors"] = int(slot_errors[mask].sum())
        row["availability_slots"] = int(mask.sum() * available.shape[1])
        row["interval_deviation_sum"] = float(deviation[mask].sum())
        row["interval_deviation_nodes"] = int((mask & (deviation > INTERVAL_TOLERANCE)).sum())
        by_k[str(int(value))] = row
    totals = {
        "states": int(uncolored.sum()), "participation_correct": int(flags["participation_correct"].sum()),
        "candidate_correct": int(flags["candidate_correct"].sum()), "forbidden_offers": int(flags["forbidden"].sum()),
        "commit_correct_own": int(flags["commit_correct"].sum()), "action_agree": int(flags["action_agree"].sum()),
        "unsafe_commits": int(flags["unsafe"].sum()), "fixed_neighbor_conflicts": int(flags["fixed_conflict_executed"].sum()),
        "simultaneous_conflicts": int(flags["simultaneous_conflict_executed"].sum()),
        "valid_commits_rejected": int(flags["unnecessary_wait"].sum()),
        "availability_slot_errors": int(slot_errors.sum()), "availability_slots": int(uncolored.sum() * available.shape[1]),
        "interval_deviation_sum": float(deviation.sum()),
        "interval_deviation_nodes": int((deviation > INTERVAL_TOLERANCE).sum()),
    }
    return totals, by_k


@torch.inference_mode()
def run_rollout(model, n, edge_index, palette, tape_seed, max_rounds=256, diagnostics=True,
                edge_chunk="auto", permutation=None):
    """Learned closed-loop execution on one graph and tape.

    ``permutation`` relabels node v as permutation[v] (edges must already be
    relabelled); the tape draws stay attached to the original nodes, so a
    node-order-equivariant policy yields the permuted coloring.
    """
    model.eval()
    device = next(model.parameters()).device
    if edge_chunk == "auto":
        edge_chunk = auto_edge_chunk(edge_index.shape[1], palette, model.message1[0].out_features)
    colors = np.full(n, -1, np.int64)
    rng = np.random.default_rng(tape_seed)
    edges_t = torch.from_numpy(np.ascontiguousarray(edge_index)).to(device)
    palette_t = torch.full((n,), palette, dtype=torch.long, device=device)
    totals, by_k = defaultdict(int), defaultdict(lambda: defaultdict(int))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    inference_seconds, first_divergence = 0.0, None
    started = time.perf_counter()
    for round_i in range(1, max_rounds + 1):
        tick = time.perf_counter()
        rp, rc = rng.random(n), rng.random(n)
        if permutation is not None:
            rp_new, rc_new = np.empty_like(rp), np.empty_like(rc)
            rp_new[permutation], rc_new[permutation] = rp, rc
            rp, rc = rp_new, rc_new
        out = model(torch.from_numpy(colors).to(device), torch.from_numpy(rp.astype(np.float32)).to(device),
                    torch.from_numpy(rc.astype(np.float32)).to(device), edges_t, palette_t, edge_chunk)
        p, c = out.participate.cpu().numpy(), out.candidate.cpu().numpy()
        commit = (out.commit_logit > 0).cpu().numpy()
        executed = (colors < 0) & p & (c >= 0) & commit
        next_colors = colors.copy()
        next_colors[executed] = c[executed]
        inference_seconds += time.perf_counter() - tick
        if diagnostics:
            observed, observed_k = observe(colors, edge_index, palette, rp, rc, p, c, commit, out.availability.cpu().numpy(),
                                           out.lo.cpu().numpy(), out.hi.cpu().numpy())
            for key, value in observed.items():
                totals[key] += value
            for value, row in observed_k.items():
                for key, count in row.items():
                    by_k[value][key] += count
            if first_divergence is None and observed["action_agree"] != observed["states"]:
                first_divergence = round_i
        colors = next_colors
        if np.all(colors >= 0):
            break
    src, dst = edge_index
    upper = src < dst
    bad_edges = int(((colors[src] >= 0) & (colors[src] == colors[dst]) & upper).sum())
    finished = bool(np.all(colors >= 0))
    result = {"finished": finished, "success": bool(finished and bad_edges == 0), "monochromatic_edges": bad_edges,
              "rounds": round_i, "uncolored_nodes": int((colors < 0).sum()), "inference_seconds": inference_seconds,
              "wall_seconds": time.perf_counter() - started, "edge_chunk": edge_chunk or 0, **dict(totals),
              "by_k": {k: dict(v) for k, v in by_k.items()}}
    if diagnostics:
        result["teacher_identical_trajectory"] = first_divergence is None
        result["first_divergence_round"] = first_divergence or 0
    if device.type == "cuda":
        result["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["gpu_peak_above_baseline_bytes"] = int(torch.cuda.max_memory_allocated(device) - baseline)
    return result, colors


def cohort_cases(degree, sizes, counts, split="final_test", namespace=20261014, families=("REGULAR", "IRREGULAR")):
    """Graph/tape metadata. ``final_test`` with namespace 20261014 is the frozen
    M6B cohort, so specialists and UCGNN are compared on identical graphs/tapes."""
    for n in sizes:
        for family in families:
            for i in range(counts[n]):
                yield {"graph_id": f"{split}/d{degree}/{family}/n{n}/{i}", "split": split, "namespace": namespace,
                       "graph_seed": seed_for(namespace, degree, f"{split}/graph", n, family, i),
                       "tape_seed": seed_for(namespace, degree, f"{split}/tape", n, family, i),
                       "n": n, "degree": degree, "distribution": family}


def build_case(metadata, cache_dir=None):
    """Edges and structural metadata of a cohort graph, cached as .npz."""
    from pathlib import Path
    from dcl.study.config import digest
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{digest([metadata['graph_id'], metadata['graph_seed']])[:24]}.npz"
        if path.exists():
            saved = np.load(path, allow_pickle=False)
            info = {k: saved[k].item() for k in ("fingerprint", "degree_min", "degree_max", "degree_mean")}
            return saved["edge_index"], info
    graph = generate_graph(metadata["n"], metadata["degree"], metadata["distribution"], metadata["graph_seed"])
    edges = _edge_index(graph)
    info = {"fingerprint": fingerprint(graph), **degree_stats(graph)}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, edge_index=edges, **{k: np.asarray(v) for k, v in info.items()})
        temporary.replace(path)
    return edges, info
