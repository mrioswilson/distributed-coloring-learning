"""Paired closed-loop comparison, without shielding."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from dcl.graphs import connected_random_regular_graph
from dcl.m4_models import make_m4_model
from evaluate_m4_seed import DEVICE, run_m4


def load_model(path):
    cp = torch.load(
        path, map_location=DEVICE, weights_only=True
    )

    model = make_m4_model(
        cp["arch"],
        hidden_dim=cp["hidden_dim"],
        message_dim=cp["message_dim"],
    ).to(DEVICE)

    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=20)
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[40, 640, 2560, 10240],
    )
    args = parser.parse_args()

    m4 = load_model("data/results/m4b_seed10001_n40.pt")
    m6 = load_model("data/results/m6b_counterfactual_seed10001_n40.pt")

    print(
        f"device={DEVICE} graphs_per_size={args.graphs}",
        flush=True,
    )

    rows = []

    for n in args.sizes:
        m4_success = m6_success = 0
        m4_only = m6_only = 0
        rounds4, rounds6 = [], []
        bad4 = bad6 = 0

        for i in range(args.graphs):
            graph_seed = 30_000_000 + n * 1000 + i
            tape_seed = 40_000_000 + n * 1000 + i

            G = connected_random_regular_graph(
                n=n, degree=3, seed=graph_seed
            )

            # Both execute without a shield.
            r4 = run_m4(m4, G, tape_seed)
            r6 = run_m4(m6, G, tape_seed)

            m4_success += int(r4["success"])
            m6_success += int(r6["success"])

            m4_only += int(
                r4["success"] and not r6["success"]
            )
            m6_only += int(
                r6["success"] and not r4["success"]
            )

            rounds4.append(r4["rounds"])
            rounds6.append(r6["rounds"])

            for name, result in (("M4B", r4), ("M6B", r6)):
                colors = result["colors"]

                bad_edges = sum(
                    colors[u] >= 0 and colors[u] == colors[v]
                    for u, v in G.edges()
                )

                if name == "M4B":
                    bad4 += bad_edges
                else:
                    bad6 += bad_edges

                rows.append({
                    "n": n,
                    "graph_i": i,
                    "model": name,
                    "finished": int(result["finished"]),
                    "success": int(result["success"]),
                    "rounds": result["rounds"],
                    "monochromatic_edges": bad_edges,
                })

        print(
            f"N={n:<5} "
            f"M4B={m4_success}/{args.graphs} "
            f"M6B={m6_success}/{args.graphs} "
            f"M4_only={m4_only} "
            f"M6_only={m6_only} "
            f"bad_edges={bad4}/{bad6} "
            f"rounds={np.mean(rounds4):.2f}/"
            f"{np.mean(rounds6):.2f}",
            flush=True,
        )

    output = Path(
        "data/results/m4b_vs_counterfactual_holdout.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
