from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dcl.dataset import encode_state
from dcl.graphs import connected_random_regular_graph
from dcl.model import CommitGNN
from dcl.oracle import (
    UNCOLORED,
    apply_commit,
    is_proper_coloring,
    run_oracle,
    sample_random_trial,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = Path(
    "data/results/m1_commit_gnn.pt"
)

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

NUM_GRAPHS = 100
MAX_ROUNDS = 1000


def load_model():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model = CommitGNN(
        num_features=checkpoint["num_features"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


@torch.inference_mode()
def model_commit_decision(
    model,
    G,
    colors,
    participate,
    candidate,
):
    state = encode_state(
        G=G,
        colors=colors,
        participate=participate,
        candidate=candidate,
    )

    x = torch.from_numpy(
        state.x
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    edge_index = torch.from_numpy(
        state.edge_index
    ).to(
        DEVICE,
        dtype=torch.long,
    )

    logits = model(
        x,
        edge_index,
    )

    pred = (
        logits > 0
    ).cpu().numpy()

    # Nodes that did not participate are never
    # allowed to commit.
    commit = (
        pred
        & participate
        & (colors == UNCOLORED)
    )

    return commit


def run_m1(
    model,
    G,
    seed,
    num_colors=4,
    max_rounds=MAX_ROUNDS,
):
    n = G.number_of_nodes()

    rng = np.random.default_rng(seed)

    colors = np.full(
        n,
        UNCOLORED,
        dtype=np.int64,
    )

    for t in range(max_rounds):
        # The random part is still provided externally.
        participate, candidate = sample_random_trial(
            G=G,
            colors=colors,
            num_colors=num_colors,
            rng=rng,
        )

        # THIS is where the oracle rule has disappeared.
        # The GNN decides COMMIT / WAIT.
        commit = model_commit_decision(
            model=model,
            G=G,
            colors=colors,
            participate=participate,
            candidate=candidate,
        )

        colors = apply_commit(
            colors=colors,
            candidate=candidate,
            commit=commit,
        )

        if np.all(colors != UNCOLORED):
            return {
                "finished": True,
                "proper": is_proper_coloring(
                    G,
                    colors,
                ),
                "rounds": t + 1,
                "colors": colors,
            }

    return {
        "finished": False,
        "proper": False,
        "rounds": max_rounds,
        "colors": colors,
    }


def main():
    print("device:", DEVICE)

    model = load_model()

    rows = []

    for n in SIZES:
        print()
        print(f"N={n}")

        finished = 0
        proper = 0
        exact = 0

        model_rounds = []
        oracle_rounds = []

        for i in range(NUM_GRAPHS):
            graph_seed = (
                30_000_000
                + n * 1000
                + i
            )

            run_seed = (
                40_000_000
                + n * 1000
                + i
            )

            G = connected_random_regular_graph(
                n=n,
                degree=3,
                seed=graph_seed,
            )

            # GNN controls the execution.
            result_m1 = run_m1(
                model=model,
                G=G,
                seed=run_seed,
            )

            # Separate reference execution.
            # It does NOT affect the GNN execution.
            result_oracle = run_oracle(
                G=G,
                seed=run_seed,
                num_colors=4,
                max_rounds=MAX_ROUNDS,
                save_trace=False,
            )

            finished += int(
                result_m1["finished"]
            )

            proper += int(
                result_m1["proper"]
            )

            model_rounds.append(
                result_m1["rounds"]
            )

            oracle_rounds.append(
                result_oracle.rounds
            )

            same = (
                result_m1["finished"]
                == result_oracle.solved
                and
                result_m1["rounds"]
                == result_oracle.rounds
                and
                np.array_equal(
                    result_m1["colors"],
                    result_oracle.colors,
                )
            )

            exact += int(same)

        mean_model = float(
            np.mean(model_rounds)
        )

        mean_oracle = float(
            np.mean(oracle_rounds)
        )

        row = {
            "n": n,
            "graphs": NUM_GRAPHS,
            "finished_rate": (
                finished / NUM_GRAPHS
            ),
            "proper_rate": (
                proper / NUM_GRAPHS
            ),
            "exact_oracle_rate": (
                exact / NUM_GRAPHS
            ),
            "mean_model_rounds": mean_model,
            "mean_oracle_rounds": mean_oracle,
            "mean_round_difference": (
                mean_model - mean_oracle
            ),
        }

        rows.append(row)

        print(
            f"finished={finished}/{NUM_GRAPHS} "
            f"proper={proper}/{NUM_GRAPHS} "
            f"exact_oracle={exact}/{NUM_GRAPHS} "
            f"model_rounds={mean_model:.3f} "
            f"oracle_rounds={mean_oracle:.3f}"
        )

    df = pd.DataFrame(rows)

    output = Path(
        "data/results/m1_extrapolation_closed_loop.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print()
    print(df.to_string(index=False))
    print()
    print("saved:", output)


if __name__ == "__main__":
    main()
