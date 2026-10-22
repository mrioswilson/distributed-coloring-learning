"""Mechanistic diagnostic of the learned candidate rule (read-only, post hoc).

For teacher states, the candidate of node v depends on its own r_c only through
the scorer. Sweeping r_c over a grid gives the learned partition of [0, 1) into
colors, which is compared with the teacher's exact partition into k intervals
of width 1/k. The misassigned fraction of the grid estimates the expected
candidate error for that state under uniform r_c.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dcl.m4_data import _edge_index
from dcl.oracle import run_oracle
from dcl.study.config import seed_for
from dcl.study.graphs import generate_graph
from .rules import available_matrix


@torch.inference_mode()
def sweep_choices(model, colors, edges, palette, grid, chunk=200):
    """choices[v, g]: learned candidate of v when every node draws r_c = grid[g]."""
    n = len(colors)
    out = []
    for start in range(0, len(grid), chunk):
        values = grid[start:start + chunk]
        copies = len(values)
        big_colors = torch.as_tensor(np.tile(colors, copies), dtype=torch.long)
        big_edges = torch.as_tensor(np.concatenate([edges + g * n for g in range(copies)], axis=1), dtype=torch.long)
        rc = torch.as_tensor(np.repeat(values, n), dtype=torch.float32)
        palette_t = torch.full((n * copies,), palette, dtype=torch.long)
        _, scores, *_ = model.stage1(big_colors, torch.zeros_like(rc), rc, big_edges, palette_t)
        out.append(scores.argmax(1).reshape(copies, n).T.numpy())
    return np.concatenate(out, axis=1)


def node_partition_errors(choices, available, grid):
    """Per uncolored node: misassigned grid fraction, forbidden fraction, threshold errors."""
    k = available.sum(1)
    ranks = available.cumsum(1) - 1
    teacher_rank = np.floor(grid[None, :] * k[:, None]).astype(int)
    position = np.where(available, ranks, -1)
    chosen_rank = np.take_along_axis(position, choices, axis=1)        # -1 if forbidden
    wrong = chosen_rank != teacher_rank
    rows = []
    for v in range(len(k)):
        changes = np.flatnonzero(np.diff(chosen_rank[v]) != 0)
        thresholds = (grid[changes] + grid[changes + 1]) / 2
        monotone = bool(np.all(np.diff(chosen_rank[v]) >= 0)) and len(changes) == k[v] - 1 and chosen_rank[v, 0] == 0
        exact = np.arange(1, k[v]) / k[v]
        error = np.abs(thresholds - exact) if monotone else np.array([np.nan])
        rows.append({"k": int(k[v]), "misassigned": float(wrong[v].mean()), "forbidden": float((chosen_rank[v] < 0).mean()),
                     "monotone_partition": monotone,
                     "max_threshold_error": float(np.max(error)) if len(error) else 0.0,
                     "mean_threshold_error": float(np.mean(error)) if len(error) else 0.0,
                     "signed_threshold_error": float(np.mean(thresholds - exact)) if monotone and len(exact) else 0.0})
    return rows


def candidate_sweep(model, degrees, graphs=10, n=160, namespace=4242, grid_points=1001, max_states=None):
    grid = (np.arange(grid_points) + 0.5) / grid_points
    rows = []
    for degree in degrees:
        palette = degree + 1
        states = 0
        for i in range(graphs):
            family = "REGULAR" if i % 2 == 0 else "IRREGULAR"
            graph = generate_graph(n, degree, family, seed_for(namespace, degree, "diagnostic/graph", n, family, i))
            edges = _edge_index(graph)
            trace = run_oracle(graph, seed_for(namespace, degree, "diagnostic/tape", n, family, i), num_colors=palette).trace
            for t in trace:
                if max_states is not None and states >= max_states:
                    break
                uncolored = np.flatnonzero(t.colors_before < 0)
                choices = sweep_choices(model, t.colors_before, edges, palette, grid)[uncolored]
                available = available_matrix(t.colors_before, edges, palette)[uncolored]
                for row in node_partition_errors(choices, available, grid):
                    rows.append({"degree": degree, "graph_i": i, "family": family, "round": t.round_idx, **row})
                states += 1
    return pd.DataFrame(rows)


def summarize_sweep(frame):
    grouped = frame.groupby(["degree", "k"]).agg(
        nodes=("misassigned", "size"), expected_candidate_error=("misassigned", "mean"),
        forbidden_fraction=("forbidden", "mean"), monotone_partitions=("monotone_partition", "mean"),
        mean_threshold_error=("mean_threshold_error", "mean"), max_threshold_error=("max_threshold_error", "max"),
        signed_threshold_error=("signed_threshold_error", "mean")).reset_index()
    overall = frame.groupby("degree").agg(nodes=("misassigned", "size"), expected_candidate_error=("misassigned", "mean"),
                                          forbidden_fraction=("forbidden", "mean")).reset_index()
    return grouped, overall
