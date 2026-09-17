from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from dcl.dataset import encode_round
from dcl.graphs import connected_random_regular_graph
from dcl.model import CommitGNN
from dcl.oracle import run_oracle


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

NUM_GRAPHS = 100

CHECKPOINT = Path(
    "data/results/m1_commit_gnn.pt"
)


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

    print(
        "checkpoint trained on N =",
        checkpoint["train_n"],
    )
    print(
        "validation accuracy =",
        checkpoint["val_accuracy"],
    )

    return model


@torch.inference_mode()
def evaluate_round(model, example):
    x = torch.from_numpy(
        example.x
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    edge_index = torch.from_numpy(
        example.edge_index
    ).to(
        DEVICE,
        dtype=torch.long,
    )

    y = torch.from_numpy(
        example.y
    ).to(
        DEVICE,
        dtype=torch.bool,
    )

    mask = torch.from_numpy(
        example.decision_mask
    ).to(
        DEVICE,
        dtype=torch.bool,
    )

    # It is possible that nobody participates
    # in a particular round.
    if not bool(mask.any()):
        return 0, 0, 0, 0, 0

    logits = model(
        x,
        edge_index,
    )

    pred = logits > 0

    pred = pred[mask]
    truth = y[mask]

    total = int(truth.numel())

    errors = int(
        (pred != truth).sum().item()
    )

    tp = int(
        (pred & truth).sum().item()
    )

    fp = int(
        (pred & ~truth).sum().item()
    )

    fn = int(
        (~pred & truth).sum().item()
    )

    return total, errors, tp, fp, fn


def main():
    print("device:", DEVICE)

    model = load_model()

    rows = []

    for n in SIZES:
        print()
        print(f"N={n}")

        total = 0
        errors = 0
        tp = 0
        fp = 0
        fn = 0

        exact_graphs = 0

        for i in range(NUM_GRAPHS):
            # These seeds are different from both
            # training and validation seeds.
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

            assert result.solved

            graph_errors = 0

            for trace in result.trace:
                example = encode_round(
                    G,
                    trace,
                )

                (
                    round_total,
                    round_errors,
                    round_tp,
                    round_fp,
                    round_fn,
                ) = evaluate_round(
                    model,
                    example,
                )

                total += round_total
                errors += round_errors
                tp += round_tp
                fp += round_fp
                fn += round_fn

                graph_errors += round_errors

            if graph_errors == 0:
                exact_graphs += 1

        accuracy = (
            1 - errors / total
            if total
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        exact_graph_rate = (
            exact_graphs / NUM_GRAPHS
        )

        row = {
            "n": n,
            "graphs": NUM_GRAPHS,
            "decisions": total,
            "errors": errors,
            "false_positives": fp,
            "false_negatives": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "exact_graph_rate": exact_graph_rate,
        }

        rows.append(row)

        print(
            f"decisions={total:,} "
            f"errors={errors:,} "
            f"accuracy={accuracy:.8f} "
            f"precision={precision:.8f} "
            f"recall={recall:.8f} "
            f"exact_graphs={exact_graphs}/{NUM_GRAPHS}"
        )

    df = pd.DataFrame(rows)

    output = Path(
        "data/results/m1_extrapolation_open_loop.csv"
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
