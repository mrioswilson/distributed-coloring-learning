from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_m4_seed import (
    load_model,
    open_loop_graph,
    predict,
)

from dcl.graphs import connected_random_regular_graph
from dcl.m4_data import encode_m4_state
from dcl.oracle import (
    UNCOLORED,
    apply_commit,
    is_proper_coloring,
    run_oracle,
    sample_random_trial_details,
)


SEEDS = [
    1234,
    2001,
    3001,
    4001,
    5001,
    6001,
    7001,
    8001,
    9001,
    10001,
]

MAX_ROUNDS = 1000


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--n",
        type=int,
        default=10240,
    )

    p.add_argument(
        "--graphs",
        type=int,
        default=100,
    )

    return p.parse_args()


def unsafe_commits(
    G,
    colors,
    candidate,
    commit,
):
    """
    Inspect the actions that are ABOUT TO BE applied.

    fixed_conflict_nodes:
        a committing node chooses the same color
        as an already-colored neighbor.

    simultaneous_conflict_edges:
        adjacent uncolored nodes commit to the
        same color in this round.
    """

    fixed_conflict_nodes = set()
    simultaneous_conflict_edges = 0

    committing = np.flatnonzero(commit)

    for v in committing:
        cv = candidate[v]

        for u in G.neighbors(v):
            if (
                colors[u] != UNCOLORED
                and colors[u] == cv
            ):
                fixed_conflict_nodes.add(
                    int(v)
                )
                break

    for u, v in G.edges():
        if (
            commit[u]
            and commit[v]
            and candidate[u] == candidate[v]
        ):
            simultaneous_conflict_edges += 1

    return (
        len(fixed_conflict_nodes),
        simultaneous_conflict_edges,
    )


def run_diagnostic(
    model,
    G,
    seed,
):
    n = G.number_of_nodes()

    colors = np.full(
        n,
        UNCOLORED,
        dtype=np.int64,
    )

    rng = np.random.default_rng(seed)

    total_fixed = 0
    total_simultaneous = 0
    first_unsafe_round = None
    total_commits = 0

    for t in range(MAX_ROUNDS):
        rnd = sample_random_trial_details(
            G=G,
            colors=colors,
            num_colors=4,
            rng=rng,
        )

        state = encode_m4_state(
            G=G,
            colors=colors,
            activation_draw=rnd.activation_draw,
            candidate_draw=rnd.candidate_draw,
        )

        (
            participate,
            candidate_class,
            commit_pred,
        ) = predict(
            model,
            state,
        )

        candidate = np.where(
            candidate_class == 0,
            UNCOLORED,
            candidate_class - 1,
        ).astype(np.int64)

        commit = (
            participate
            & commit_pred
            & (colors == UNCOLORED)
            & (candidate != UNCOLORED)
        )

        fixed, simultaneous = unsafe_commits(
            G,
            colors,
            candidate,
            commit,
        )

        total_fixed += fixed
        total_simultaneous += simultaneous
        total_commits += int(commit.sum())

        if (
            first_unsafe_round is None
            and (fixed > 0 or simultaneous > 0)
        ):
            first_unsafe_round = t + 1

        colors = apply_commit(
            colors=colors,
            candidate=candidate,
            commit=commit,
        )

        if np.all(colors != UNCOLORED):
            return {
                "finished": True,
                "proper": bool(
                    is_proper_coloring(
                        G,
                        colors,
                    )
                ),
                "rounds": t + 1,
                "fixed_conflict_nodes":
                    total_fixed,
                "simultaneous_conflict_edges":
                    total_simultaneous,
                "unsafe_events":
                    total_fixed
                    + total_simultaneous,
                "first_unsafe_round":
                    first_unsafe_round,
                "total_commits":
                    total_commits,
            }

    return {
        "finished": False,
        "proper": False,
        "rounds": MAX_ROUNDS,
        "fixed_conflict_nodes":
            total_fixed,
        "simultaneous_conflict_edges":
            total_simultaneous,
        "unsafe_events":
            total_fixed
            + total_simultaneous,
        "first_unsafe_round":
            first_unsafe_round,
        "total_commits":
            total_commits,
    }


def main():
    args = parse_args()

    rows = []

    for seed in SEEDS:
        print()
        print("=" * 60)
        print(
            f"M4B seed={seed} N={args.n}"
        )
        print("=" * 60)

        model = load_model(
            "B",
            seed,
        )

        for i in range(args.graphs):
            graph_seed = (
                10_000_000
                + args.n * 1000
                + i
            )

            run_seed = (
                20_000_000
                + args.n * 1000
                + i
            )

            G = connected_random_regular_graph(
                n=args.n,
                degree=3,
                seed=graph_seed,
            )

            oracle = run_oracle(
                G=G,
                seed=run_seed,
                num_colors=4,
                max_rounds=MAX_ROUNDS,
                save_trace=True,
            )

            open_stats = open_loop_graph(
                model,
                G,
                oracle,
            )

            closed = run_diagnostic(
                model,
                G,
                run_seed,
            )

            rows.append(
                {
                    "seed": seed,
                    "graph_id": i,
                    "n": args.n,

                    "success":
                        int(
                            closed["finished"]
                            and closed["proper"]
                        ),

                    "finished":
                        int(closed["finished"]),

                    "rounds":
                        closed["rounds"],

                    "unsafe_events":
                        closed["unsafe_events"],

                    "fixed_conflict_nodes":
                        closed[
                            "fixed_conflict_nodes"
                        ],

                    "simultaneous_conflict_edges":
                        closed[
                            "simultaneous_conflict_edges"
                        ],

                    "first_unsafe_round":
                        closed[
                            "first_unsafe_round"
                        ],

                    "total_commits":
                        closed["total_commits"],

                    "open_errors":
                        open_stats["errors"],

                    "open_false_commit":
                        open_stats["false_commit"],

                    "open_missed_commit":
                        open_stats["missed_commit"],

                    "open_wrong_color":
                        open_stats["wrong_color"],
                }
            )

            if (i + 1) % 10 == 0:
                done = pd.DataFrame(rows)

                current = done[
                    done["seed"] == seed
                ]

                print(
                    f"{i + 1:3d}/{args.graphs} "
                    f"success="
                    f"{current['success'].mean():.3f}"
                )

    df = pd.DataFrame(rows)

    output = (
        Path("data/results")
        / f"m4b_failure_modes_n{args.n}.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print()
    print("PER-SEED")
    print()

    per_seed = (
        df.groupby("seed")
        .agg(
            success_rate=(
                "success",
                "mean",
            ),
            unsafe_mean=(
                "unsafe_events",
                "mean",
            ),
            fixed_mean=(
                "fixed_conflict_nodes",
                "mean",
            ),
            simultaneous_mean=(
                "simultaneous_conflict_edges",
                "mean",
            ),
            open_false_commit_mean=(
                "open_false_commit",
                "mean",
            ),
            open_missed_commit_mean=(
                "open_missed_commit",
                "mean",
            ),
            open_wrong_color_mean=(
                "open_wrong_color",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        per_seed.to_string(
            index=False
        )
    )

    print()
    print("CLOSED-LOOP SAFETY CHECK")
    print()

    table = pd.crosstab(
        df["unsafe_events"] > 0,
        df["success"],
        rownames=["has_unsafe_commit"],
        colnames=["success"],
    )

    print(table)

    print()
    print("OPEN-LOOP ERRORS: SUCCESS VS FAILURE")
    print()

    comparison = (
        df.groupby("success")[
            [
                "open_false_commit",
                "open_missed_commit",
                "open_wrong_color",
                "open_errors",
            ]
        ]
        .mean()
    )

    print(comparison)

    print()
    print(
        "CORRELATION WITH SUCCESS "
        "(pooled per graph)"
    )
    print()

    for col in [
        "open_false_commit",
        "open_missed_commit",
        "open_wrong_color",
        "open_errors",
        "unsafe_events",
        "fixed_conflict_nodes",
        "simultaneous_conflict_edges",
    ]:
        corr = df[
            ["success", col]
        ].corr().iloc[0, 1]

        print(
            f"{col:30s} {corr: .6f}"
        )

    print()
    print("saved:", output)


if __name__ == "__main__":
    main()
