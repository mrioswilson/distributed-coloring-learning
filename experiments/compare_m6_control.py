"""Paired M4B / oracle-control / DAgger evaluation."""

import argparse
import csv
from pathlib import Path

import numpy as np

from dcl.graphs import connected_random_regular_graph
from compare_m4_m6 import load_model
from evaluate_m4_seed import run_m4


N = 10240

CHECKPOINTS = {
    "M4B": "data/results/m4b_seed1234_n40.pt",
    "CONTROL": "data/results/m4b_seed1234_extra_oracle_control.pt",
    "DAGGER": "data/results/m6b_seed1234_iter1.pt",
}

OUTPUT = Path(
    "data/results/m6_three_way_n10240_seed1234.csv"
)

FIELDS = ["graph_i"]

for name in CHECKPOINTS:
    FIELDS.extend([
        f"{name}_success",
        f"{name}_finished",
        f"{name}_rounds",
        f"{name}_bad_edges",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graphs", type=int, default=100
    )
    args = parser.parse_args()

    models = {
        name: load_model(path)
        for name, path in CHECKPOINTS.items()
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    if OUTPUT.exists():
        with OUTPUT.open(newline="") as f:
            results = {
                int(row["graph_i"]): row
                for row in csv.DictReader(f)
            }

    for i in range(args.graphs):
        if i in results:
            continue

        graph_seed = 10_000_000 + N * 1000 + i
        tape_seed = 20_000_000 + N * 1000 + i

        G = connected_random_regular_graph(
            n=N, degree=3, seed=graph_seed
        )

        row = {"graph_i": i}

        for name, model in models.items():
            # NO SHIELD: genuine learned execution.
            result = run_m4(
                model, G, tape_seed
            )

            colors = result["colors"]

            bad_edges = sum(
                colors[u] >= 0
                and colors[u] == colors[v]
                for u, v in G.edges()
            )

            row[f"{name}_success"] = int(
                result["success"]
            )
            row[f"{name}_finished"] = int(
                result["finished"]
            )
            row[f"{name}_rounds"] = result["rounds"]
            row[f"{name}_bad_edges"] = bad_edges

        # Persist immediately; safe to resume.
        new_file = not OUTPUT.exists()

        with OUTPUT.open("a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=FIELDS
            )

            if new_file:
                writer.writeheader()

            writer.writerow(row)

        results[i] = row

        if (i + 1) % 10 == 0:
            print(
                f"Completed {i + 1}/{args.graphs}",
                flush=True,
            )

    rows = [
        results[i]
        for i in range(args.graphs)
        if i in results
    ]

    print()
    print(f"RESULTS N={N} graphs={len(rows)}")

    for name in CHECKPOINTS:
        success = sum(
            int(r[f"{name}_success"])
            for r in rows
        )
        bad = sum(
            int(r[f"{name}_bad_edges"])
            for r in rows
        )
        rounds = np.mean([
            float(r[f"{name}_rounds"])
            for r in rows
        ])

        print(
            f"{name:8} "
            f"success={success}/{len(rows)} "
            f"bad_edges={bad} "
            f"rounds={rounds:.3f}"
        )

    print()
    print("PAIRED COMPARISONS")

    for a, b in [
        ("DAGGER", "M4B"),
        ("DAGGER", "CONTROL"),
        ("CONTROL", "M4B"),
    ]:
        a_only = sum(
            int(r[f"{a}_success"]) == 1
            and int(r[f"{b}_success"]) == 0
            for r in rows
        )

        b_only = sum(
            int(r[f"{a}_success"]) == 0
            and int(r[f"{b}_success"]) == 1
            for r in rows
        )

        print(
            f"{a} vs {b}: "
            f"{a}_only={a_only} "
            f"{b}_only={b_only}"
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
