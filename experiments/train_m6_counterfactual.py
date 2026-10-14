"""Train full M4B from scratch with counterfactual supervision."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dcl.counterfactual import (
    commit_logits_for_proposals,
    exact_commit_labels,
    make_counterfactual_proposals,
)
from dcl.graphs import connected_random_regular_graph
from dcl.m4_data import encode_m4_round
from dcl.m4_models import make_m4_model
from dcl.oracle import run_oracle
from dcl.staged_models import count_parameters

from train_m4_seed import (
    DEVICE,
    evaluate,
    generate_examples,
    make_batch,
)


def collect_training_examples(graphs, tapes):
    """
    Repeat Luby on the SAME training graphs,
    with independent random tapes.
    """
    examples = []

    for i in range(graphs):
        graph = connected_random_regular_graph(
            n=40,
            degree=3,
            seed=100_000 + i,
        )

        for k in range(tapes):
            result = run_oracle(
                graph,
                seed=200_000 + i + 1_000_000 * k,
                num_colors=4,
                save_trace=True,
            )

            if not result.solved:
                raise RuntimeError("Oracle failed")

            examples.extend(
                encode_m4_round(graph, trace)
                for trace in result.trace
            )

    return examples


def train_step(
    model, optimizer, examples, epoch, warmup, rng
):
    """
    Train all three M4B heads jointly.

    COMMIT learns from the actual proposals that
    are supplied to its communication pathway.
    """
    x, edges, yp, yc, ycommit, mask = make_batch(
        examples
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Original M4B forward.
    p_logits, c_logits, own_commit_logits = model(
        x, edges, hard_messages=True
    )

    p_loss = F.binary_cross_entropy_with_logits(
        p_logits[mask], yp[mask]
    )

    c_loss = F.cross_entropy(
        c_logits[mask], yc[mask]
    )

    if epoch <= warmup:
        # All three heads are trained from epoch 1.
        # Initially COMMIT sees the correct oracle
        # proposals and their matching labels.
        expert_logits = commit_logits_for_proposals(
            model, x, edges, yp > 0.5, yc
        )

        commit_loss = (
            F.binary_cross_entropy_with_logits(
                expert_logits[mask],
                ycommit[mask],
            )
        )

        own_loss = commit_loss
        cf_loss = commit_loss

    else:
        # 1. The GNN produces its OWN proposals.
        with torch.no_grad():
            p_pred = (p_logits > 0).cpu().numpy()

            c_pred = (
                c_logits.argmax(dim=1)
                .cpu().numpy()
            )

            x_np = x.cpu().numpy()
            e_np = edges.cpu().numpy()

            target_own_np = exact_commit_labels(
                x_np, e_np, p_pred, c_pred
            )

        target_own = torch.as_tensor(
            target_own_np, device=DEVICE
        )

        own_loss = (
            F.binary_cross_entropy_with_logits(
                own_commit_logits[mask],
                target_own[mask],
            )
        )

        # 2. Generate deliberately altered proposals.
        with torch.no_grad():
            p_cf, c_cf = make_counterfactual_proposals(
                x_np,
                e_np,
                yp.cpu().numpy(),
                yc.cpu().numpy(),
                rng,
            )

            target_cf = exact_commit_labels(
                x_np, e_np, p_cf, c_cf
            )

        # 3. Feed those proposals into M4B's
        #    unchanged COMMIT communication pathway.
        cf_logits = commit_logits_for_proposals(
            model,
            x,
            edges,
            torch.as_tensor(p_cf, device=DEVICE),
            torch.as_tensor(c_cf, device=DEVICE),
        )

        target_cf = torch.as_tensor(
            target_cf, device=DEVICE
        )

        cf_loss = (
            F.binary_cross_entropy_with_logits(
                cf_logits[mask],
                target_cf[mask],
            )
        )

        commit_loss = 0.5 * (
            own_loss + cf_loss
        )

    # All three heads remain trainable.
    loss = p_loss + c_loss + commit_loss

    loss.backward()
    optimizer.step()

    return (
        float(loss.item()),
        float(p_loss.item()),
        float(c_loss.item()),
        float(own_loss.item()),
        float(cf_loss.item()),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=10001
    )
    parser.add_argument(
        "--epochs", type=int, default=120
    )
    parser.add_argument(
        "--train-graphs", type=int, default=800
    )
    parser.add_argument(
        "--val-graphs", type=int, default=200
    )
    parser.add_argument(
        "--tapes-per-graph", type=int, default=3
    )
    parser.add_argument(
        "--batch-size", type=int, default=128
    )
    parser.add_argument(
        "--warmup", type=int, default=10
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3
    )
    parser.add_argument(
        "--smoke", action="store_true"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = np.random.default_rng(
        args.seed + 17
    )

    print(
        "device:", DEVICE,
        "seed:", args.seed,
        flush=True,
    )

    print(
        "Generating training and validation data...",
        flush=True,
    )

    train = collect_training_examples(
        args.train_graphs,
        args.tapes_per_graph,
    )

    val = generate_examples(
        args.val_graphs,
        300_000,
        400_000,
    )

    print(
        "training rounds:", len(train),
        "validation rounds:", len(val),
        flush=True,
    )

    # Fresh random initialization.
    model = make_m4_model(
        "B",
        hidden_dim=64,
        message_dim=32,
    ).to(DEVICE)

    print(
        "parameters:", count_parameters(model),
        flush=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    suffix = "_smoke" if args.smoke else ""

    output = Path(
        f"data/results/"
        f"m6b_counterfactual_seed{args.seed}"
        f"_n40{suffix}.pt"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best = None

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)

        running = []

        for start in range(
            0, len(train), args.batch_size
        ):
            batch = train[
                start:start + args.batch_size
            ]

            metrics = train_step(
                model,
                optimizer,
                batch,
                epoch,
                args.warmup,
                rng,
            )

            running.append(metrics)

        # Standard M4B validation, without shielding.
        metrics = evaluate(model, val)

        score = (
            metrics["action_acc"],
            metrics["joint_acc"],
            -metrics["loss"],
        )

        losses = np.asarray(
            running
        ).mean(axis=0)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={losses[0]:.5f} "
            f"part_loss={losses[1]:.5f} "
            f"cand_loss={losses[2]:.5f} "
            f"commit_own={losses[3]:.5f} "
            f"commit_cf={losses[4]:.5f} "
            f"val_action={metrics['action_acc']:.6f} "
            f"val_joint={metrics['joint_acc']:.6f}",
            flush=True,
        )

        if best is None or score > best:
            best = score

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch": "B",
                    "hidden_dim": 64,
                    "message_dim": 32,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_metrics": metrics,
                    "method": "M6B-counterfactual-from-scratch",
                    "train_graphs": args.train_graphs,
                    "tapes_per_graph": args.tapes_per_graph,
                    "warmup": args.warmup,
                },
                output,
            )

    print("BEST:", best, flush=True)
    print("SAVED:", output, flush=True)


if __name__ == "__main__":
    main()
