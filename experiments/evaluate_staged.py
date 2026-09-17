from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dcl.graphs import (
    connected_random_regular_graph,
)
from dcl.oracle import (
    UNCOLORED,
    apply_commit,
    is_proper_coloring,
    run_oracle,
    sample_random_trial_details,
)
from dcl.staged_data import (
    encode_m2_round,
    encode_m2_state,
    encode_m3_round,
    encode_m3_state,
)
from dcl.staged_models import make_model


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
        "--task",
        choices=["m2", "m3"],
        required=True,
    )

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


def to_tensors(state):
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

    return x, edge_index


def load_model(
    task,
    arch,
):
    path = Path(
        "data/results"
    ) / (
        f"{task}{arch.lower()}"
        "_n40.pt"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model = make_model(
        task=task,
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
def open_loop_graph(
    model,
    task,
    G,
    result,
):
    stage1_total = 0
    stage1_errors = 0

    commit_total = 0
    commit_errors = 0

    fp = 0
    fn = 0

    for trace in result.trace:
        if task == "m2":
            ex = encode_m2_round(
                G,
                trace,
            )
        else:
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

        mask = torch.from_numpy(
            ex.decision_mask
        ).to(
            DEVICE,
            dtype=torch.bool,
        )

        out1, commit_logit = model(
            x,
            edge_index,
            hard_messages=True,
        )

        if task == "m2":
            truth1 = torch.from_numpy(
                ex.y_participate
            ).to(
                DEVICE,
                dtype=torch.bool,
            )

            pred1 = (
                out1 > 0
            )

        else:
            truth1 = torch.from_numpy(
                ex.y_candidate
            ).to(
                DEVICE,
                dtype=torch.long,
            )

            pred1 = (
                out1.argmax(dim=1)
            )

        truth_commit = torch.from_numpy(
            ex.y_commit
        ).to(
            DEVICE,
            dtype=torch.bool,
        )

        pred_commit = (
            commit_logit > 0
        )

        stage1_total += int(
            mask.sum().item()
        )

        commit_total += int(
            mask.sum().item()
        )

        stage1_errors += int(
            (
                pred1[mask]
                != truth1[mask]
            ).sum().item()
        )

        commit_errors += int(
            (
                pred_commit[mask]
                != truth_commit[mask]
            ).sum().item()
        )

        fp += int(
            (
                pred_commit[mask]
                & ~truth_commit[mask]
            ).sum().item()
        )

        fn += int(
            (
                ~pred_commit[mask]
                & truth_commit[mask]
            ).sum().item()
        )

    return {
        "stage1_total": stage1_total,
        "stage1_errors": stage1_errors,
        "commit_total": commit_total,
        "commit_errors": commit_errors,
        "fp": fp,
        "fn": fn,
    }


@torch.inference_mode()
def run_learned(
    model,
    task,
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
        sample = sample_random_trial_details(
            G=G,
            colors=colors,
            num_colors=4,
            rng=rng,
        )

        if task == "m2":
            state = encode_m2_state(
                G=G,
                colors=colors,
                activation_draw=(
                    sample.activation_draw
                ),
                candidate_offer=(
                    sample.candidate_offer
                ),
            )

            x, edge_index = to_tensors(
                state
            )

            (
                participation_logit,
                commit_logit,
            ) = model(
                x,
                edge_index,
                hard_messages=True,
            )

            predicted_participation = (
                participation_logit > 0
            ).cpu().numpy()

            predicted_commit = (
                commit_logit > 0
            ).cpu().numpy()

            commit = (
                predicted_participation
                & predicted_commit
                & (
                    colors
                    == UNCOLORED
                )
                & (
                    sample.candidate_offer
                    != UNCOLORED
                )
            )

            candidate = (
                sample.candidate_offer
            )

        else:
            state = encode_m3_state(
                G=G,
                colors=colors,
                participate=(
                    sample.participate
                ),
                candidate_draw=(
                    sample.candidate_draw
                ),
            )

            x, edge_index = to_tensors(
                state
            )

            (
                candidate_logits,
                commit_logit,
            ) = model(
                x,
                edge_index,
                hard_messages=True,
            )

            candidate_class = (
                candidate_logits
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

            candidate = np.where(
                candidate_class == 0,
                UNCOLORED,
                candidate_class - 1,
            ).astype(
                np.int64
            )

            predicted_commit = (
                commit_logit > 0
            ).cpu().numpy()

            commit = (
                predicted_commit
                & sample.participate
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
            return {
                "finished": True,
                "proper":
                    is_proper_coloring(
                        G,
                        colors,
                    ),
                "rounds": t + 1,
                "colors": colors,
            }

    return {
        "finished": False,
        "proper": False,
        "rounds": MAX_ROUNDS,
        "colors": colors,
    }


def main():
    args = parse_args()

    print("device:", DEVICE)

    model = load_model(
        args.task,
        args.arch,
    )

    rows = []

    for n in SIZES:
        open_stage1_total = 0
        open_stage1_errors = 0

        open_commit_total = 0
        open_commit_errors = 0

        open_fp = 0
        open_fn = 0

        open_exact_graphs = 0

        finished = 0
        proper = 0
        exact_oracle = 0

        learned_rounds = []
        oracle_rounds = []

        print()
        print(f"N={n}")

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
                model=model,
                task=args.task,
                G=G,
                result=oracle,
            )

            open_stage1_total += (
                stats["stage1_total"]
            )

            open_stage1_errors += (
                stats["stage1_errors"]
            )

            open_commit_total += (
                stats["commit_total"]
            )

            open_commit_errors += (
                stats["commit_errors"]
            )

            open_fp += stats["fp"]
            open_fn += stats["fn"]

            if (
                stats["stage1_errors"] == 0
                and
                stats["commit_errors"] == 0
            ):
                open_exact_graphs += 1

            learned = run_learned(
                model=model,
                task=args.task,
                G=G,
                seed=run_seed,
            )

            finished += int(
                learned["finished"]
            )

            proper += int(
                learned["proper"]
            )

            learned_rounds.append(
                learned["rounds"]
            )

            oracle_rounds.append(
                oracle.rounds
            )

            same = (
                learned["finished"]
                == oracle.solved
                and
                learned["rounds"]
                == oracle.rounds
                and
                np.array_equal(
                    learned["colors"],
                    oracle.colors,
                )
            )

            exact_oracle += int(same)

        stage1_acc = (
            1
            - open_stage1_errors
            / open_stage1_total
        )

        commit_acc = (
            1
            - open_commit_errors
            / open_commit_total
        )

        row = {
            "task":
                args.task,
            "arch":
                args.arch,
            "n":
                n,
            "graphs":
                args.graphs,
            "open_stage1_decisions":
                open_stage1_total,
            "open_stage1_errors":
                open_stage1_errors,
            "open_stage1_acc":
                stage1_acc,
            "open_commit_errors":
                open_commit_errors,
            "open_commit_acc":
                commit_acc,
            "open_commit_fp":
                open_fp,
            "open_commit_fn":
                open_fn,
            "open_exact_graph_rate":
                open_exact_graphs
                / args.graphs,
            "closed_finished_rate":
                finished
                / args.graphs,
            "closed_proper_rate":
                proper
                / args.graphs,
            "closed_exact_oracle_rate":
                exact_oracle
                / args.graphs,
            "mean_learned_rounds":
                float(
                    np.mean(
                        learned_rounds
                    )
                ),
            "mean_oracle_rounds":
                float(
                    np.mean(
                        oracle_rounds
                    )
                ),
        }

        rows.append(row)

        print(
            f"stage1_acc={stage1_acc:.8f} "
            f"stage1_errors={open_stage1_errors:,} "
            f"commit_acc={commit_acc:.8f} "
            f"commit_errors={open_commit_errors:,} "
            f"FP={open_fp:,} "
            f"FN={open_fn:,} "
            f"open_exact={open_exact_graphs}/{args.graphs} "
            f"proper={proper}/{args.graphs} "
            f"closed_exact={exact_oracle}/{args.graphs}"
        )

    df = pd.DataFrame(rows)

    output = Path(
        "data/results"
    ) / (
        f"{args.task}{args.arch.lower()}"
        "_extrapolation.csv"
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
