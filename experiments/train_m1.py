from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dcl.dataset import encode_round
from dcl.graphs import connected_random_regular_graph
from dcl.model import CommitGNN
from dcl.oracle import run_oracle


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_GRAPHS = 800
VAL_GRAPHS = 200

BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 1e-3


def generate_examples(
    num_graphs: int,
    graph_seed_base: int,
    run_seed_base: int,
):
    examples = []

    for i in range(num_graphs):
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
                encode_round(G, trace)
            )

    return examples


def make_batch(examples):
    xs = []
    ys = []
    masks = []
    edge_indices = []

    offset = 0

    for example in examples:
        n = example.x.shape[0]

        xs.append(torch.from_numpy(example.x))
        ys.append(torch.from_numpy(example.y))
        masks.append(torch.from_numpy(example.decision_mask))

        edge_indices.append(
            torch.from_numpy(example.edge_index) + offset
        )

        offset += n

    x = torch.cat(xs).to(
        DEVICE,
        dtype=torch.float32,
    )

    y = torch.cat(ys).to(
        DEVICE,
        dtype=torch.float32,
    )

    mask = torch.cat(masks).to(
        DEVICE,
        dtype=torch.bool,
    )

    edge_index = torch.cat(
        edge_indices,
        dim=1,
    ).to(
        DEVICE,
        dtype=torch.long,
    )

    return x, edge_index, y, mask


@torch.no_grad()
def evaluate(model, examples):
    model.eval()

    correct = 0
    total = 0

    tp = 0
    fp = 0
    fn = 0

    total_loss = 0.0
    batches = 0

    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start:start + BATCH_SIZE]

        x, edge_index, y, mask = make_batch(batch)

        logits = model(x, edge_index)

        loss = F.binary_cross_entropy_with_logits(
            logits[mask],
            y[mask],
        )

        total_loss += loss.item()
        batches += 1

        pred = logits[mask] > 0
        truth = y[mask] > 0.5

        correct += int((pred == truth).sum())
        total += int(truth.numel())

        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())

    accuracy = correct / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "loss": total_loss / batches,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "decisions": total,
    }


def main():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    print("device:", DEVICE)

    print("Generating training data...")
    train_examples = generate_examples(
        num_graphs=TRAIN_GRAPHS,
        graph_seed_base=1_000_000,
        run_seed_base=2_000_000,
    )

    print("Generating validation data...")
    val_examples = generate_examples(
        num_graphs=VAL_GRAPHS,
        graph_seed_base=3_000_000,
        run_seed_base=4_000_000,
    )

    train_decisions = sum(
        int(e.decision_mask.sum())
        for e in train_examples
    )

    train_commits = sum(
        int(e.y[e.decision_mask].sum())
        for e in train_examples
    )

    print("training rounds:", len(train_examples))
    print("validation rounds:", len(val_examples))
    print("training decisions:", train_decisions)
    print(
        "commit fraction:",
        train_commits / train_decisions,
    )

    model = CommitGNN(
        num_features=11,
        hidden_dim=32,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    shuffle_rng = random.Random(1234)

    best_val_accuracy = -1.0
    output = Path("data/results/m1_commit_gnn.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        indices = list(range(len(train_examples)))
        shuffle_rng.shuffle(indices)

        epoch_loss = 0.0
        batches = 0

        for start in range(0, len(indices), BATCH_SIZE):
            ids = indices[start:start + BATCH_SIZE]
            batch = [train_examples[i] for i in ids]

            x, edge_index, y, mask = make_batch(batch)

            optimizer.zero_grad()

            logits = model(x, edge_index)

            loss = F.binary_cross_entropy_with_logits(
                logits[mask],
                y[mask],
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        val = evaluate(model, val_examples)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={epoch_loss / batches:.5f} "
            f"val_loss={val['loss']:.5f} "
            f"val_acc={val['accuracy']:.5f} "
            f"precision={val['precision']:.5f} "
            f"recall={val['recall']:.5f}"
        )

        if val["accuracy"] > best_val_accuracy:
            best_val_accuracy = val["accuracy"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_features": 11,
                    "hidden_dim": 32,
                    "train_n": 40,
                    "seed": 1234,
                    "val_accuracy": best_val_accuracy,
                },
                output,
            )

    print()
    print("best validation accuracy:", best_val_accuracy)
    print("saved:", output)


if __name__ == "__main__":
    main()
