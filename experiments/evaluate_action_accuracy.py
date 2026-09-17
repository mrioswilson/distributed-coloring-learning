from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dcl.graphs import connected_random_regular_graph
from dcl.oracle import UNCOLORED, run_oracle
from dcl.staged_data import encode_m2_round, encode_m3_round
from dcl.staged_models import make_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
        default=100,
    )

    return parser.parse_args()


def load_model(task, arch):
    path = (
        Path("data/results")
        / f"{task}{arch.lower()}_n40.pt"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model = make_model(
        task=task,
        arch=arch,
        hidden_dim=checkpoint["hidden_dim"],
        message_dim=checkpoint["message_dim"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("checkpoint:", path)
    print("parameters:", checkpoint["parameters"])

    return model


@torch.inference_mode()
def evaluate_graph(model, task, G, result):
    total = 0
    errors = 0

    false_commit = 0
    missed_commit = 0
    wrong_color = 0

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

        out1, commit_logit = model(
            x,
            edge_index,
            hard_messages=True,
        )

        n = G.number_of_nodes()

        # Action codes:
        #
        # 0 = WAIT
        # 1 = COMMIT color 0
        # 2 = COMMIT color 1
        # 3 = COMMIT color 2
        # 4 = COMMIT color 3

        truth_action = np.zeros(
            n,
            dtype=np.int64,
        )

        commit_truth = trace.commit

        truth_action[commit_truth] = (
            trace.candidate[commit_truth] + 1
        )

        if task == "m2":
            predicted_participation = (
                out1 > 0
            ).cpu().numpy()

            predicted_commit = (
                commit_logit > 0
            ).cpu().numpy()

            action_commit = (
                predicted_participation
                & predicted_commit
                & (
                    trace.colors_before
                    == UNCOLORED
                )
                & (
                    trace.candidate_offer
                    != UNCOLORED
                )
            )

            pred_action = np.zeros(
                n,
                dtype=np.int64,
            )

            pred_action[action_commit] = (
                trace.candidate_offer[
                    action_commit
                ]
                + 1
            )

        else:
            candidate_class = (
                out1.argmax(dim=1)
                .cpu()
                .numpy()
            )

            predicted_commit = (
                commit_logit > 0
            ).cpu().numpy()

            action_commit = (
                trace.participate
                & predicted_commit
                & (
                    candidate_class != 0
                )
                & (
                    trace.colors_before
                    == UNCOLORED
                )
            )

            pred_action = np.zeros(
                n,
                dtype=np.int64,
            )

            pred_action[action_commit] = (
                candidate_class[
                    action_commit
                ]
            )

        mask = (
            trace.colors_before
            == UNCOLORED
        )

        truth = truth_action[mask]
        pred = pred_action[mask]

        total += len(truth)

        errors += int(
            np.sum(
                pred != truth
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
        "errors": errors,
        "false_commit": false_commit,
        "missed_commit": missed_commit,
        "wrong_color": wrong_color,
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
        total = 0
        errors = 0
        false_commit = 0
        missed_commit = 0
        wrong_color = 0

        exact_graphs = 0

        for i in range(args.graphs):
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

            stats = evaluate_graph(
                model,
                args.task,
                G,
                result,
            )

            total += stats["total"]
            errors += stats["errors"]

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
                exact_graphs += 1

        accuracy = (
            1.0 - errors / total
        )

        print(
            f"N={n:<5} "
            f"action_acc={accuracy:.8f} "
            f"errors={errors:,} "
            f"false_commit={false_commit:,} "
            f"missed_commit={missed_commit:,} "
            f"wrong_color={wrong_color:,} "
            f"exact={exact_graphs}/{args.graphs}"
        )

        rows.append(
            {
                "task": args.task,
                "arch": args.arch,
                "n": n,
                "decisions": total,
                "action_errors": errors,
                "action_accuracy": accuracy,
                "false_commit": false_commit,
                "missed_commit": missed_commit,
                "wrong_color": wrong_color,
                "exact_graph_rate":
                    exact_graphs
                    / args.graphs,
            }
        )

    df = pd.DataFrame(rows)

    output = (
        Path("data/results")
        / (
            f"{args.task}"
            f"{args.arch.lower()}"
            "_action_accuracy.csv"
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
