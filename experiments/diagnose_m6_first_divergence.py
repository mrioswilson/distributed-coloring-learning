"""Audit first differences between CONTROL and COUNTERFACTUAL."""

import csv
from pathlib import Path

import numpy as np

from dcl.graphs import connected_random_regular_graph
from dcl.counterfactual import exact_commit_labels
from compare_m4_m6 import load_model
from evaluate_m4_seed import run_m4


N = 10240
BASE = Path("data/results")

CHECKPOINTS = {
    "CONTROL": BASE / "m6b_ownonly_seed10001_n40.pt",
    "COUNTERFACTUAL": BASE / "m6b_counterfactual_seed10001_n40.pt",
}


def trace_execution(model, graph, tape_seed):
    """Observe the original execution without changing its actions."""
    trace = []
    edges = None

    def hook(module, inputs, outputs):
        nonlocal edges

        x, edge_index = inputs[:2]
        p_logits, c_logits, k_logits = outputs

        if edges is None:
            edges = edge_index.detach().cpu().numpy().copy()

        trace.append({
            "x": x.detach().cpu().numpy().copy(),
            "p": (p_logits.detach().cpu().numpy() > 0),
            "c": c_logits.detach().argmax(1).cpu().numpy(),
            "k": (k_logits.detach().cpu().numpy() > 0),
        })

    handle = model.register_forward_hook(hook)

    try:
        result = run_m4(model, graph, tape_seed)
    finally:
        handle.remove()

    return result, trace, edges


def actions_and_targets(snapshot, edges):
    x = snapshot["x"]
    p = snapshot["p"]
    c = snapshot["c"]
    k = snapshot["k"]

    uncolored = x[:, 0] > 0.5

    # Predicted effective commit, without shield.
    actions = uncolored & p & (c != 0) & k

    # Exact mathematical correctness for these proposals.
    targets = exact_commit_labels(
        x, edges, p, c
    ).astype(bool)

    return actions, targets


def verify_trace(result, trace, edges):
    """Check that reconstructed actions match run_m4."""
    for r, snapshot in enumerate(trace):
        actions, _ = actions_and_targets(snapshot, edges)

        classes = snapshot["x"][:, :5].argmax(1).copy()
        classes[actions] = snapshot["c"][actions]

        if r + 1 < len(trace):
            next_classes = trace[r + 1]["x"][:, :5].argmax(1)

            if not np.array_equal(classes, next_classes):
                raise RuntimeError(
                    f"Reconstructed execution differs at round {r + 1}. "
                    "Inspect run_m4 before interpreting this diagnosis."
                )
        else:
            if not np.array_equal(
                classes - 1, np.asarray(result["colors"])
            ):
                raise RuntimeError(
                    "Reconstructed final coloring differs from run_m4."
                )


def reason(graph, snapshot, v):
    x = snapshot["x"]
    p = snapshot["p"]
    c = snapshot["c"]

    color = int(c[v])

    if not p[v] or color == 0:
        return "inactive_or_no_candidate"

    uncolored = x[:, 0] > 0.5
    fixed_class = x[:, :5].argmax(1)

    neighbors = list(graph.neighbors(v))

    if any(
        not uncolored[u] and fixed_class[u] == color
        for u in neighbors
    ):
        return "fixed_neighbor_conflict"

    if any(
        uncolored[u] and p[u] and c[u] == color
        for u in neighbors
    ):
        return "simultaneous_conflict"

    return "valid_proposal"


def first_unsafe(graph, trace, edges):
    for r, snapshot in enumerate(trace, start=1):
        actions, targets = actions_and_targets(
            snapshot, edges
        )

        unsafe = np.flatnonzero(actions & ~targets)

        if len(unsafe):
            v = int(unsafe[0])

            return {
                "round": r,
                "node": v,
                "reason": reason(graph, snapshot, v),
            }

    return None


def first_difference(graph, trace_a, trace_b, edges):
    for r, (a, b) in enumerate(
        zip(trace_a, trace_b), start=1
    ):
        if not np.array_equal(a["x"], b["x"]):
            raise RuntimeError(
                f"States diverged before an effective commit "
                f"difference was detected (round {r})."
            )

        if (
            not np.array_equal(a["p"], b["p"])
            or not np.array_equal(a["c"], b["c"])
        ):
            raise RuntimeError(
                f"PARTICIPATE/CANDIDATE differ on "
                f"the same state at round {r}."
            )

        act_a, targets = actions_and_targets(a, edges)
        act_b, _ = actions_and_targets(b, edges)

        differing = np.flatnonzero(act_a != act_b)

        if len(differing):
            rows = []

            for v in differing:
                v = int(v)

                rows.append({
                    "round": r,
                    "node": v,
                    "proposed_color": int(a["c"][v]) - 1,
                    "rp": float(a["x"][v, -2]),
                    "rc": float(a["x"][v, -1]),
                    "control_commit": int(act_a[v]),
                    "counterfactual_commit": int(act_b[v]),
                    "correct_commit": int(targets[v]),
                    "reason": reason(graph, a, v),
                })

            return rows

    return []


def save_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    path = BASE / "m6_overnight_three_way_seed10001.csv"

    with path.open(newline="") as f:
        evaluated = list(csv.DictReader(f))

    failures = [
        row for row in evaluated
        if int(row["CONTROL_success"]) == 0
        and int(row["COUNTERFACTUAL_success"]) == 1
    ]

    print("Control-only failures:", len(failures), flush=True)

    models = {
        name: load_model(str(path))
        for name, path in CHECKPOINTS.items()
    }

    details = []
    summaries = []

    for row in failures:
        i = int(row["graph_i"])

        graph_seed = 50_000_000 + N * 1000 + i
        tape_seed = 60_000_000 + N * 1000 + i

        graph = connected_random_regular_graph(
            n=N, degree=3, seed=graph_seed
        )

        result_a, trace_a, edges_a = trace_execution(
            models["CONTROL"], graph, tape_seed
        )

        result_b, trace_b, edges_b = trace_execution(
            models["COUNTERFACTUAL"], graph, tape_seed
        )

        verify_trace(result_a, trace_a, edges_a)
        verify_trace(result_b, trace_b, edges_b)

        if not np.array_equal(edges_a, edges_b):
            raise RuntimeError("Different edge encodings")

        if int(result_a["success"]) != 0:
            raise RuntimeError(f"Control result mismatch: graph {i}")

        if int(result_b["success"]) != 1:
            raise RuntimeError(f"Counterfactual result mismatch: graph {i}")

        differences = first_difference(
            graph, trace_a, trace_b, edges_a
        )

        unsafe_a = first_unsafe(
            graph, trace_a, edges_a
        )

        unsafe_b = first_unsafe(
            graph, trace_b, edges_b
        )

        for item in differences:
            details.append({
                "graph_i": i,
                **item,
            })

        summaries.append({
            "graph_i": i,
            "first_diff_round": (
                differences[0]["round"] if differences else ""
            ),
            "first_diff_nodes": len(differences),
            "first_diff_reason": (
                differences[0]["reason"] if differences else ""
            ),
            "control_first_unsafe_round": (
                unsafe_a["round"] if unsafe_a else ""
            ),
            "control_first_unsafe_node": (
                unsafe_a["node"] if unsafe_a else ""
            ),
            "control_first_unsafe_reason": (
                unsafe_a["reason"] if unsafe_a else ""
            ),
            "counterfactual_first_unsafe_round": (
                unsafe_b["round"] if unsafe_b else ""
            ),
        })

        print(
            f"graph={i:3d} "
            f"first_diff={differences[0]['round'] if differences else 'NONE'} "
            f"nodes={len(differences)} "
            f"control_unsafe={unsafe_a} "
            f"counterfactual_unsafe={unsafe_b}",
            flush=True,
        )

    summary_path = BASE / "m6_first_divergence_summary.csv"
    detail_path = BASE / "m6_first_divergence_nodes.csv"

    save_csv(
        summary_path,
        summaries,
        [
            "graph_i",
            "first_diff_round",
            "first_diff_nodes",
            "first_diff_reason",
            "control_first_unsafe_round",
            "control_first_unsafe_node",
            "control_first_unsafe_reason",
            "counterfactual_first_unsafe_round",
        ],
    )

    save_csv(
        detail_path,
        details,
        [
            "graph_i",
            "round",
            "node",
            "proposed_color",
            "rp",
            "rc",
            "control_commit",
            "counterfactual_commit",
            "correct_commit",
            "reason",
        ],
    )

    print()
    print("Saved:", summary_path)
    print("Saved:", detail_path)


if __name__ == "__main__":
    main()
