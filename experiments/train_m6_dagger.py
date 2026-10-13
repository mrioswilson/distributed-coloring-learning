"""M6: first DAgger training-step verification."""

import argparse
import random

import torch
import torch.nn.functional as F

from dcl.dagger import collect_dagger_rollout
from dcl.graphs import connected_random_regular_graph
from dcl.m4_data import encode_m4_round
from dcl.m4_models import make_m4_model
from dcl.oracle import run_oracle

from train_m4_seed import DEVICE, make_batch


def dagger_train_step(
    model, optimizer, original, dagger,
    batch_size=16, rng=None,
):
    """One balanced M4B training minibatch."""

    if not original or not dagger:
        raise ValueError("Both datasets must be nonempty")

    if batch_size < 2:
        raise ValueError("batch_size must be >= 2")

    rng = rng or random
    n_old = batch_size // 2

    batch = (
        rng.choices(original, k=n_old)
        + rng.choices(dagger, k=batch_size - n_old)
    )
    rng.shuffle(batch)

    x, edges, yp, yc, ycommit, mask = make_batch(batch)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    p, c, commit = model(
        x, edges, hard_messages=True
    )

    # Exactly the same post-warmup loss as M4B.
    loss = (
        F.binary_cross_entropy_with_logits(
            p[mask], yp[mask]
        )
        + F.cross_entropy(
            c[mask], yc[mask]
        )
        + F.binary_cross_entropy_with_logits(
            commit[mask], ycommit[mask]
        )
    )

    loss.backward()
    optimizer.step()

    return float(loss.item())


def smoke_test():
    torch.manual_seed(123)

    model = make_m4_model(
        "B", hidden_dim=8, message_dim=4
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )

    G = connected_random_regular_graph(
        40, degree=3, seed=50
    )

    teacher_run = run_oracle(
        G, seed=60, num_colors=4,
        save_trace=True,
    )

    original = [
        encode_m4_round(G, trace)
        for trace in teacher_run.trace
    ]

    rollout = collect_dagger_rollout(
        G, model, seed=70,
        max_rounds=8, device=DEVICE,
    )

    if not rollout.examples:
        raise AssertionError("No DAgger examples collected")

    before = [
        p.detach().clone()
        for p in model.parameters()
    ]

    loss = dagger_train_step(
        model, optimizer,
        original, rollout.examples,
    )

    changed = any(
        not torch.equal(p, old)
        for p, old in zip(
            model.parameters(), before
        )
    )

    assert changed
    assert torch.isfinite(torch.tensor(loss))

    print(f"device={DEVICE}")
    print(f"oracle_examples={len(original)}")
    print(f"dagger_examples={len(rollout.examples)}")
    print(f"loss={loss:.6f}")
    print(f"weights_changed={changed}")
    print("M6 training-step smoke test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not args.smoke:
        parser.error("Use --smoke for this first step")

    smoke_test()
