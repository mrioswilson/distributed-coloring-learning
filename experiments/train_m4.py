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
from dcl.m4_data import (
    encode_m4_round,
)
from dcl.m4_models import (
    make_m4_model,
)
from dcl.staged_models import (
    count_parameters,
)
from dcl.oracle import run_oracle


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
EPOCHS = 120
WARMUP = 10


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arch",
        choices=["A", "B"],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    return parser.parse_args()


def generate_examples(
    count,
    graph_seed_base,
    run_seed_base,
):
    examples = []

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
                encode_m4_round(
                    G,
                    trace,
                )
            )

    return examples


def make_batch(examples):
    xs = []
    edges = []

    yp = []
    yc = []
    ycommit = []
    masks = []

    offset = 0

    for ex in examples:
        n = ex.x.shape[0]

        xs.append(ex.x)

        edges.append(
            ex.edge_index + offset
        )

        yp.append(
            ex.y_participate
        )

        yc.append(
            ex.y_candidate_offer
        )

        ycommit.append(
            ex.y_commit
        )

        masks.append(
            ex.decision_mask
        )

        offset += n

    x = torch.from_numpy(
        np.concatenate(xs)
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

    yp = torch.from_numpy(
        np.concatenate(yp)
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    yc = torch.from_numpy(
        np.concatenate(yc)
    ).to(
        DEVICE,
        dtype=torch.long,
    )

    ycommit = torch.from_numpy(
        np.concatenate(ycommit)
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    mask = torch.from_numpy(
        np.concatenate(masks)
    ).to(
        DEVICE,
        dtype=torch.bool,
    )

    return (
        x,
        edge_index,
        yp,
        yc,
        ycommit,
        mask,
    )


@torch.inference_mode()
def evaluate(
    model,
    examples,
):
    model.eval()

    total = 0

    p_correct = 0
    c_correct = 0
    commit_correct = 0
    joint_correct = 0

    action_correct = 0

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
            yp,
            yc,
            ycommit,
            mask,
        ) = make_batch(batch)

        (
            p_logit,
            c_logits,
            commit_logit,
        ) = model(
            x,
            edge_index,
            hard_messages=True,
        )

        n = int(
            mask.sum().item()
        )

        if n == 0:
            continue

        p_loss = (
            F.binary_cross_entropy_with_logits(
                p_logit[mask],
                yp[mask],
            )
        )

        c_loss = F.cross_entropy(
            c_logits[mask],
            yc[mask],
        )

        commit_loss = (
            F.binary_cross_entropy_with_logits(
                commit_logit[mask],
                ycommit[mask],
            )
        )

        loss = (
            p_loss
            + c_loss
            + commit_loss
        )

        p_pred = (
            p_logit[mask] > 0
        )

        p_truth = (
            yp[mask] > 0.5
        )

        c_pred = (
            c_logits[mask]
            .argmax(dim=1)
        )

        c_truth = yc[mask]

        commit_pred = (
            commit_logit[mask] > 0
        )

        commit_truth = (
            ycommit[mask] > 0.5
        )

        p_ok = (
            p_pred == p_truth
        )

        c_ok = (
            c_pred == c_truth
        )

        commit_ok = (
            commit_pred
            == commit_truth
        )

        # Final action:
        # 0 = WAIT
        # 1..4 = COMMIT color 0..3

        pred_action = torch.zeros(
            n,
            device=DEVICE,
            dtype=torch.long,
        )

        pred_do_commit = (
            p_pred
            & commit_pred
            & (c_pred != 0)
        )

        pred_action[
            pred_do_commit
        ] = c_pred[
            pred_do_commit
        ]

        truth_action = torch.zeros(
            n,
            device=DEVICE,
            dtype=torch.long,
        )

        truth_action[
            commit_truth
        ] = c_truth[
            commit_truth
        ]

        total += n

        p_correct += int(
            p_ok.sum().item()
        )

        c_correct += int(
            c_ok.sum().item()
        )

        commit_correct += int(
            commit_ok.sum().item()
        )

        joint_correct += int(
            (
                p_ok
                & c_ok
                & commit_ok
            ).sum().item()
        )

        action_correct += int(
            (
                pred_action
                == truth_action
            ).sum().item()
        )

        loss_sum += (
            float(loss.item())
            * n
        )

    return {
        "loss":
            loss_sum / total,

        "participation_acc":
            p_correct / total,

        "candidate_acc":
            c_correct / total,

        "commit_acc":
            commit_correct / total,

        "joint_acc":
            joint_correct / total,

        "action_acc":
            action_correct / total,
    }


def main():
    args = parse_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    print("device:", DEVICE)
    print("M4 architecture:", args.arch)

    print(
        "Generating training data..."
    )

    train = generate_examples(
        count=TRAIN_GRAPHS,
        graph_seed_base=100_000,
        run_seed_base=200_000,
    )

    print(
        "Generating validation data..."
    )

    val = generate_examples(
        count=VAL_GRAPHS,
        graph_seed_base=300_000,
        run_seed_base=400_000,
    )

    model = make_m4_model(
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

    output = (
        Path("data/results")
        / f"m4{args.arch.lower()}_n40.pt"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_score = None

    for epoch in range(
        1,
        args.epochs + 1,
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
                yp,
                yc,
                ycommit,
                mask,
            ) = make_batch(batch)

            optimizer.zero_grad()

            hard = (
                epoch > WARMUP
            )

            (
                p_logit,
                c_logits,
                commit_logit,
            ) = model(
                x,
                edge_index,
                hard_messages=hard,
            )

            p_loss = (
                F.binary_cross_entropy_with_logits(
                    p_logit[mask],
                    yp[mask],
                )
            )

            c_loss = F.cross_entropy(
                c_logits[mask],
                yc[mask],
            )

            # First learn the two ingredients.
            if epoch <= WARMUP:
                loss = (
                    p_loss
                    + c_loss
                )

            else:
                commit_loss = (
                    F.binary_cross_entropy_with_logits(
                        commit_logit[mask],
                        ycommit[mask],
                    )
                )

                loss = (
                    p_loss
                    + c_loss
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
        )

        train_loss = (
            train_loss_sum
            / train_nodes
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={metrics['loss']:.5f} "
            f"part={metrics['participation_acc']:.6f} "
            f"candidate={metrics['candidate_acc']:.6f} "
            f"commit={metrics['commit_acc']:.6f} "
            f"joint={metrics['joint_acc']:.6f} "
            f"action={metrics['action_acc']:.6f}"
        )

        if epoch <= WARMUP:
            continue

        # The real task is the final action.
        score = (
            metrics["action_acc"],
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

    print()
    print(
        "best score:",
        best_score,
    )

    print(
        "saved:",
        output,
    )


if __name__ == "__main__":
    main()
