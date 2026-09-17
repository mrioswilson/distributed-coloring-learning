from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dcl.graphs import connected_random_regular_graph
from dcl.oracle import (
    UNCOLORED,
    available_colors,
    run_oracle,
)
from dcl.staged_data import encode_m3_round
from dcl.staged_models import make_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arch",
        choices=["A", "B"],
        required=True,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10240,
    )

    parser.add_argument(
        "--graphs",
        type=int,
        default=100,
    )

    return parser.parse_args()


def load_model(arch):
    path = (
        Path("data/results")
        / f"m3{arch.lower()}_n40.pt"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model = make_model(
        task="m3",
        arch=arch,
        hidden_dim=checkpoint["hidden_dim"],
        message_dim=checkpoint["message_dim"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def distance_to_threshold(
    r: float,
    palette_size: int,
) -> float:
    if palette_size <= 1:
        return float("inf")

    thresholds = [
        j / palette_size
        for j in range(
            1,
            palette_size
        )
    ]

    return min(
        abs(r - t)
        for t in thresholds
    )


@torch.inference_mode()
def main():
    args = parse_args()

    model = load_model(
        args.arch
    )

    bins = [
        0.0,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        float("inf"),
    ]

    bucket_total = defaultdict(int)
    bucket_errors = defaultdict(int)

    palette_total = defaultdict(int)
    palette_errors = defaultdict(int)

    total = 0
    errors = 0

    error_distances = []

    for i in range(args.graphs):
        n = args.n

        graph_seed = (
            10_000_000
            + n * 1000
            + i
        )

        run_seed = (
            20_000_000
            + n * 1000
            + i
        )

        G = connected_random_regular_graph(
            n=n,
            degree=3,
            seed=graph_seed,
        )

        result = run_oracle(
            G=G,
            seed=run_seed,
            num_colors=4,
            save_trace=True,
        )

        for trace in result.trace:
            ex = encode_m3_round(
                G,
                trace,
            )

            x = torch.from_numpy(
                ex.x
            ).to(
                DEVICE,
                dtype=torch.float32,
            )

            edge_index = torch.from_numpy(
                ex.edge_index
            ).to(
                DEVICE,
                dtype=torch.long,
            )

            candidate_logits, _ = model(
                x,
                edge_index,
                hard_messages=True,
            )

            pred = (
                candidate_logits
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

            for v in range(n):
                # Candidate selection only exists
                # for participating uncolored nodes.
                if (
                    trace.colors_before[v]
                    != UNCOLORED
                ):
                    continue

                if not trace.participate[v]:
                    continue

                truth = (
                    int(trace.candidate[v])
                    + 1
                )

                palette = available_colors(
                    G=G,
                    colors=trace.colors_before,
                    num_colors=4,
                    v=v,
                )

                k = len(palette)

                r = float(
                    trace.candidate_draw[v]
                )

                d = distance_to_threshold(
                    r,
                    k,
                )

                total += 1
                palette_total[k] += 1

                label = None

                for lo, hi in zip(
                    bins[:-1],
                    bins[1:],
                ):
                    if lo <= d < hi:
                        label = (
                            f"[{lo:g},{hi:g})"
                        )
                        break

                if label is None:
                    label = "inf"

                bucket_total[label] += 1

                if pred[v] != truth:
                    errors += 1

                    palette_errors[k] += 1
                    bucket_errors[label] += 1

                    if np.isfinite(d):
                        error_distances.append(d)

    print()
    print(
        f"M3{args.arch} "
        f"N={args.n} "
        f"graphs={args.graphs}"
    )

    print()
    print(
        "candidate decisions:",
        f"{total:,}",
    )

    print(
        "candidate errors:",
        f"{errors:,}",
    )

    print(
        "error rate:",
        f"{errors / total:.8f}",
    )

    print()
    print("BY PALETTE SIZE")
    print()

    for k in sorted(
        palette_total
    ):
        t = palette_total[k]
        e = palette_errors[k]

        print(
            f"k={k}: "
            f"total={t:,} "
            f"errors={e:,} "
            f"error_rate={e/t:.8f}"
        )

    print()
    print("BY DISTANCE TO NEAREST THRESHOLD")
    print()

    for lo, hi in zip(
        bins[:-1],
        bins[1:],
    ):
        label = (
            f"[{lo:g},{hi:g})"
        )

        t = bucket_total[label]
        e = bucket_errors[label]

        if t == 0:
            continue

        print(
            f"{label:16s} "
            f"total={t:9,d} "
            f"errors={e:7,d} "
            f"error_rate={e/t:.8f}"
        )

    if error_distances:
        arr = np.asarray(
            error_distances
        )

        print()
        print("ERROR DISTANCES")
        print()

        print(
            "median:",
            float(
                np.median(arr)
            ),
        )

        print(
            "p90:",
            float(
                np.quantile(
                    arr,
                    0.90,
                )
            ),
        )

        for threshold in [
            0.001,
            0.0025,
            0.005,
            0.01,
            0.02,
            0.05,
        ]:
            fraction = float(
                np.mean(
                    arr < threshold
                )
            )

            print(
                f"errors within "
                f"{threshold:g}: "
                f"{fraction:.4%}"
            )


if __name__ == "__main__":
    main()
