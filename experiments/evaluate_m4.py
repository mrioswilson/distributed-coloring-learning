from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dcl.graphs import (
    connected_random_regular_graph,
)
from dcl.m4_data import (
    encode_m4_round,
    encode_m4_state,
)
from dcl.m4_models import (
    make_m4_model,
)
from dcl.oracle import (
    UNCOLORED,
    apply_commit,
    is_proper_coloring,
    run_oracle,
    sample_random_trial_details,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arch",
        choices=["A", "B"],
        required=True,
    )

    parser.add_argument(
        "--graphs",
        type=int,
        default=NUM_GRAPHS,
    )

    return parser.parse_args()


def load_model(arch):
    path = (
        Path("data/results")
        / f"m4{arch.lower()}_n40.pt"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model = make_m4_model(
        arch=arch,
        hidden_dim=checkpoint[
            "hidden_dim"
        ],
        message_dim=checkpoint[
            "message_dim"
        ],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "checkpoint:",
        path,
    )

    print(
        "parameters:",
        checkpoint["parameters"],
    )

    print(
        "validation:",
        checkpoint["val_metrics"],
    )

    return model


@torch.inference_mode()
def predict(
    model,
    state,
):
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

    p, c, commit = model(
        x,
        edge_index,
        hard_messages=True,
    )

    return (
        (p > 0).cpu().numpy(),
        c.argmax(
            dim=1
        ).cpu().numpy(),
        (commit > 0).cpu().numpy(),
    )


@torch.inference_mode()
def open_loop_graph(
    model,
    G,
    result,
):
    total = 0
    action_errors = 0

    false_commit = 0
    missed_commit = 0
    wrong_color = 0

    for trace in result.trace:
        ex = encode_m4_round(
            G,
            trace,
        )

        state = type(
            "State",
            (),
            {
                "x": ex.x,
                "edge_index":
                    ex.edge_index,
            },
        )()

        (
            pred_p,
            pred_c,
            pred_commit,
        ) = predict(
            model,
            state,
        )

        mask = (
            trace.colors_before
            == UNCOLORED
        )

        truth_action = np.zeros(
            G.number_of_nodes(),
            dtype=np.int64,
        )

        truth_action[
            trace.commit
        ] = (
            trace.candidate[
                trace.commit
            ]
            + 1
        )

        pred_action = np.zeros(
            G.number_of_nodes(),
            dtype=np.int64,
        )

        do_commit = (
            pred_p
            & pred_commit
            & (pred_c != 0)
            & mask
        )

        pred_action[
            do_commit
        ] = pred_c[
            do_commit
        ]

        truth = truth_action[mask]
        pred = pred_action[mask]

        total += len(truth)

        action_errors += int(
            np.sum(
                truth != pred
            )
        )

        false_commit += int(
            np.sum(
                (truth == 0)
                & (pred != 0)
            )
        )

        missed_commit += int(
            np.sum(
                (truth != 0)
                & (pred == 0)
            )
        )

        wrong_color += int(
            np.sum(
                (truth != 0)
                & (pred != 0)
                & (truth != pred)
            )
        )

    return {
        "total": total,
        "errors": action_errors,
        "false_commit":
            false_commit,
        "missed_commit":
            missed_commit,
        "wrong_color":
            wrong_color,
    }


@torch.inference_mode()
def run_m4(
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

    rng = np.random.default_rng(
        seed
    )

    for t in range(MAX_ROUNDS):
        # We use this ONLY to obtain the two
        # random draws.
        #
        # participate/candidate/candidate_offer
        # from the oracle are ignored.
        random_sample = (
            sample_random_trial_details(
                G=G,
                colors=colors,
                num_colors=4,
                rng=rng,
            )
        )

        state = encode_m4_state(
            G=G,
            colors=colors,
            activation_draw=(
                random_sample
                .activation_draw
            ),
            candidate_draw=(
                random_sample
                .candidate_draw
            ),
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
        ).astype(
            np.int64
        )

        commit = (
            participate
            & commit_pred
            & (
                colors
                == UNCOLORED
            )
            & (
                candidate
                != UNCOLORED
            )
        )

        colors = apply_commit(
            colors=colors,
            candidate=candidate,
            commit=commit,
        )

        if np.all(
            colors != UNCOLORED
        ):
            proper = (
                is_proper_coloring(
                    G,
                    colors,
                )
            )

            return {
                "finished": True,
                "proper": proper,
                "success": proper,
                "rounds": t + 1,
                "colors": colors,
            }

    return {
        "finished": False,
        "proper": False,
        "success": False,
        "rounds": MAX_ROUNDS,
        "colors": colors,
    }


def main():
    args = parse_args()

    print("device:", DEVICE)

    model = load_model(
        args.arch
    )

    rows = []

    for n in SIZES:
        open_total = 0
        open_errors = 0

        false_commit = 0
        missed_commit = 0
        wrong_color = 0

        open_exact = 0

        finished = 0
        success = 0

        learned_rounds = []
        oracle_rounds = []

        for i in range(
            args.graphs
        ):
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

            G = (
                connected_random_regular_graph(
                    n=n,
                    degree=3,
                    seed=graph_seed,
                )
            )

            oracle = run_oracle(
                G=G,
                seed=run_seed,
                num_colors=4,
                max_rounds=MAX_ROUNDS,
                save_trace=True,
            )

            stats = open_loop_graph(
                model,
                G,
                oracle,
            )

            open_total += stats["total"]
            open_errors += stats["errors"]

            false_commit += (
                stats["false_commit"]
            )

            missed_commit += (
                stats["missed_commit"]
            )

            wrong_color += (
                stats["wrong_color"]
            )

            if stats["errors"] == 0:
                open_exact += 1

            learned = run_m4(
                model,
                G,
                run_seed,
            )

            finished += int(
                learned["finished"]
            )

            success += int(
                learned["success"]
            )

            learned_rounds.append(
                learned["rounds"]
            )

            oracle_rounds.append(
                oracle.rounds
            )

        action_acc = (
            1.0
            - open_errors
            / open_total
        )

        mean_learned = float(
            np.mean(
                learned_rounds
            )
        )

        mean_oracle = float(
            np.mean(
                oracle_rounds
            )
        )

        print()
        print(
            f"N={n:<5} "
            f"action_acc={action_acc:.8f} "
            f"errors={open_errors:,} "
            f"FC={false_commit:,} "
            f"MC={missed_commit:,} "
            f"WC={wrong_color:,} "
            f"open_exact={open_exact}/{args.graphs} "
            f"finished={finished}/{args.graphs} "
            f"SUCCESS={success}/{args.graphs} "
            f"rounds={mean_learned:.2f} "
            f"oracle={mean_oracle:.2f}"
        )

        rows.append(
            {
                "arch":
                    args.arch,

                "n":
                    n,

                "graphs":
                    args.graphs,

                "open_action_accuracy":
                    action_acc,

                "open_action_errors":
                    open_errors,

                "false_commit":
                    false_commit,

                "missed_commit":
                    missed_commit,

                "wrong_color":
                    wrong_color,

                "open_exact_graph_rate":
                    open_exact
                    / args.graphs,

                "finished_rate":
                    finished
                    / args.graphs,

                "success_rate":
                    success
                    / args.graphs,

                "mean_learned_rounds":
                    mean_learned,

                "mean_oracle_rounds":
                    mean_oracle,
            }
        )

    df = pd.DataFrame(rows)

    output = (
        Path("data/results")
        / (
            f"m4{args.arch.lower()}"
            "_extrapolation.csv"
        )
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
