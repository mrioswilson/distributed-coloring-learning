from __future__ import annotations

import pandas as pd

from dcl.graphs import connected_random_regular_graph
from dcl.oracle import run_oracle


SIZES = [
    40,
    80,
    160,
    320,
    640,
    1280,
    2560,
    5120,
    10240,
]

NUM_INSTANCES = 100


def main() -> None:
    rows = []

    for n in SIZES:
        print(f"N={n}")

        for i in range(NUM_INSTANCES):
            graph_seed = 10_000_000 + n * 1000 + i
            run_seed = 20_000_000 + n * 1000 + i

            G = connected_random_regular_graph(
                n=n,
                degree=3,
                seed=graph_seed,
            )

            result = run_oracle(
                G=G,
                seed=run_seed,
                num_colors=4,
                max_rounds=1000,
                save_trace=False,
            )

            rows.append(
                {
                    "n": n,
                    "instance": i,
                    "graph_seed": graph_seed,
                    "run_seed": run_seed,
                    "solved": result.solved,
                    "rounds": result.rounds,
                }
            )

    df = pd.DataFrame(rows)

    output = "data/results/oracle_scaling.csv"
    df.to_csv(output, index=False)

    summary = (
        df.groupby("n")
        .agg(
            success_rate=("solved", "mean"),
            mean_rounds=("rounds", "mean"),
            median_rounds=("rounds", "median"),
            p95_rounds=("rounds", lambda x: x.quantile(0.95)),
            max_rounds=("rounds", "max"),
        )
    )

    print()
    print(summary)
    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
