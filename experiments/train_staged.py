from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dcl.graphs import (
    connected_random_regular_graph,
)
from dcl.oracle import run_oracle
from dcl.staged_data import (
    encode_m2_round,
    encode_m3_round,
)
from dcl.staged_models import (
    count_parameters,
    make_model,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TRAIN_GRAPHS = 800
VAL_GRAPHS = 200

BATCH_SIZE = 128

HIDDEN_DIM = 64
MESSAGE_DIM = 32

LEARNING_RATE = 1e-3


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
        "--epochs",
        type=int,
        default=None,
    )

    return parser.parse_args()


def generate_examples(
    task,
    count,
    graph_seed_base,
    run_seed_base,
):
    examples = []

    encoder = (
        encode_m2_round
        if task == "m2"
        else encode_m3_round
    )

    for i in range(count):
        G = connected_random_regular_graph(
            n=40,
            degree=3,
            seed=graph_seed_base + i,
        )

        result = run_oracle(
            G=G,
            seed=run_seed_base + i,
            num_colors=4,
            save_trace=True,
        )

        assert result.solved

        for trace in result.trace:
            examples.append(
                encoder(
                    G,
                    trace,
                )
            )

    return examples


def make_batch(
    examples,
    task,
):
    xs = []
    edges = []
    masks = []

    offset = 0

    if task == "m2":
        y1 = []
        y2 = []
    else:
        y1 = []
        y2 = []

    for ex in examples:
        n = ex.x.shape[0]

        xs.append(ex.x)

        edges.append(
            ex.edge_index + offset
        )

        masks.append(
            ex.decision_mask
        )

        if task == "m2":
            y1.append(
                ex.y_participate
            )
            y2.append(
                ex.y_commit
            )
        else:
            y1.append(
                ex.y_candidate
            )
            y2.append(
                ex.y_commit
            )

        offset += n

    x = torch.from_numpy(
        np.concatenate(
            xs,
            axis=0,
        )
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    edge_index = torch.from_numpy(
        np.concatenate(
            edges,
            axis=1,
        )
    ).to(
        DEVICE,
        dtype=torch.long,
    )

    mask = torch.from_numpy(
        np.concatenate(
            masks,
            axis=0,
        )
    ).to(
        DEVICE,
        dtype=torch.bool,
    )

    if task == "m2":
        target1 = torch.from_numpy(
            np.concatenate(
                y1,
                axis=0,
            )
        ).to(
            DEVICE,
            dtype=torch.float32,
        )
    else:
        target1 = torch.from_numpy(
            np.concatenate(
                y1,
                axis=0,
            )
        ).to(
            DEVICE,
            dtype=torch.long,
        )

    target2 = torch.from_numpy(
        np.concatenate(
            y2,
            axis=0,
        )
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    return (
        x,
        edge_index,
        target1,
        target2,
        mask,
    )


@torch.inference_mode()
def evaluate(
    model,
    examples,
    task,
):
    model.eval()

    total = 0

    stage1_correct = 0
    commit_correct = 0
    joint_correct = 0

    commit_tp = 0
    commit_fp = 0
    commit_fn = 0

    loss_sum = 0.0

    for start in range(
        0,
        len(examples),
        BATCH_SIZE,
    ):
        batch = examples[
            start:start + BATCH_SIZE
        ]

        (
            x,
            edge_index,
            target1,
            target2,
            mask,
        ) = make_batch(
            batch,
            task,
        )

        out1, commit_logit = model(
            x,
            edge_index,
            hard_messages=True,
        )

        n = int(mask.sum().item())

        if n == 0:
            continue

        if task == "m2":
            stage1_loss = (
                F.binary_cross_entropy_with_logits(
                    out1[mask],
                    target1[mask],
                )
            )

            pred1 = (
                out1[mask] > 0
            )

            truth1 = (
                target1[mask] > 0.5
            )

        else:
            stage1_loss = F.cross_entropy(
                out1[mask],
                target1[mask],
            )

            pred1 = (
                out1[mask]
                .argmax(dim=1)
            )

            truth1 = target1[mask]

        commit_loss = (
            F.binary_cross_entropy_with_logits(
                commit_logit[mask],
                target2[mask],
            )
        )

        loss = (
            stage1_loss
            + commit_loss
        )

        pred_commit = (
            commit_logit[mask] > 0
        )

        truth_commit = (
            target2[mask] > 0.5
        )

        stage1_ok = (
            pred1 == truth1
        )

        commit_ok = (
            pred_commit
            == truth_commit
        )

        total += n

        stage1_correct += int(
            stage1_ok.sum().item()
        )

        commit_correct += int(
            commit_ok.sum().item()
        )

        joint_correct += int(
            (
                stage1_ok
                & commit_ok
            ).sum().item()
        )

        commit_tp += int(
            (
                pred_commit
                & truth_commit
            ).sum().item()
        )

        commit_fp += int(
            (
                pred_commit
                & ~truth_commit
            ).sum().item()
        )

        commit_fn += int(
            (
                ~pred_commit
                & truth_commit
            ).sum().item()
        )

        loss_sum += (
            float(loss.item())
            * n
        )

    precision = (
        commit_tp
        / (commit_tp + commit_fp)
        if commit_tp + commit_fp
        else 0.0
    )

    recall = (
        commit_tp
        / (commit_tp + commit_fn)
        if commit_tp + commit_fn
        else 0.0
    )

    return {
        "loss": loss_sum / total,
        "stage1_acc": (
            stage1_correct / total
        ),
        "commit_acc": (
            commit_correct / total
        ),
        "joint_acc": (
            joint_correct / total
        ),
        "precision": precision,
        "recall": recall,
    }


def main():
    args = parse_args()

    epochs = args.epochs

    if epochs is None:
        epochs = (
            80
            if args.task == "m2"
            else 100
        )

    warmup = (
        5
        if args.task == "m2"
        else 8
    )

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    print("device:", DEVICE)
    print(
        "task:",
        args.task.upper(),
        args.arch,
    )

    print(
        "Generating training data..."
    )

    train = generate_examples(
        task=args.task,
        count=TRAIN_GRAPHS,
        graph_seed_base=100_000,
        run_seed_base=200_000,
    )

    print(
        "Generating validation data..."
    )

    val = generate_examples(
        task=args.task,
        count=VAL_GRAPHS,
        graph_seed_base=300_000,
        run_seed_base=400_000,
    )

    model = make_model(
        task=args.task,
        arch=args.arch,
        hidden_dim=HIDDEN_DIM,
        message_dim=MESSAGE_DIM,
    ).to(DEVICE)

    parameters = count_parameters(
        model
    )

    print(
        "training rounds:",
        len(train),
    )

    print(
        "validation rounds:",
        len(val),
    )

    print(
        "parameters:",
        parameters,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_score = None
    perfect_streak = 0

    output = Path(
        "data/results"
    ) / (
        f"{args.task}{args.arch.lower()}"
        "_n40.pt"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        random.shuffle(train)

        train_loss_sum = 0.0
        train_nodes = 0

        for start in range(
            0,
            len(train),
            BATCH_SIZE,
        ):
            batch = train[
                start:start + BATCH_SIZE
            ]

            (
                x,
                edge_index,
                target1,
                target2,
                mask,
            ) = make_batch(
                batch,
                args.task,
            )

            optimizer.zero_grad()

            hard = (
                epoch > warmup
            )

            out1, commit_logit = model(
                x,
                edge_index,
                hard_messages=hard,
            )

            if args.task == "m2":
                stage1_loss = (
                    F.binary_cross_entropy_with_logits(
                        out1[mask],
                        target1[mask],
                    )
                )
            else:
                stage1_loss = (
                    F.cross_entropy(
                        out1[mask],
                        target1[mask],
                    )
                )

            if epoch <= warmup:
                loss = stage1_loss
            else:
                commit_loss = (
                    F.binary_cross_entropy_with_logits(
                        commit_logit[mask],
                        target2[mask],
                    )
                )

                loss = (
                    stage1_loss
                    + commit_loss
                )

            loss.backward()

            optimizer.step()

            n = int(
                mask.sum().item()
            )

            train_loss_sum += (
                float(loss.item())
                * n
            )

            train_nodes += n

        metrics = evaluate(
            model,
            val,
            args.task,
        )

        train_loss = (
            train_loss_sum
            / train_nodes
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={metrics['loss']:.5f} "
            f"stage1_acc={metrics['stage1_acc']:.6f} "
            f"commit_acc={metrics['commit_acc']:.6f} "
            f"joint_acc={metrics['joint_acc']:.6f} "
            f"precision={metrics['precision']:.6f} "
            f"recall={metrics['recall']:.6f}"
        )

        if epoch <= warmup:
            continue

        score = (
            metrics["joint_acc"],
            -metrics["loss"],
        )

        if (
            best_score is None
            or score > best_score
        ):
            best_score = score

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "task":
                        args.task,
                    "arch":
                        args.arch,
                    "train_n":
                        40,
                    "hidden_dim":
                        HIDDEN_DIM,
                    "message_dim":
                        MESSAGE_DIM,
                    "parameters":
                        parameters,
                    "val_metrics":
                        metrics,
                },
                output,
            )

        if metrics["joint_acc"] == 1.0:
            perfect_streak += 1
        else:
            perfect_streak = 0

        if perfect_streak >= 5:
            print(
                "Perfect validation "
                "for 5 consecutive epochs."
            )
            break

    print()
    print("saved:", output)


if __name__ == "__main__":
    main()
