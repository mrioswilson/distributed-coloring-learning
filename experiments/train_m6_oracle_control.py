"""Extra-oracle-data control for M6 DAgger iteration 1."""

import random
from pathlib import Path

import numpy as np
import torch

from dcl.graphs import connected_random_regular_graph
from dcl.oracle import run_oracle
from dcl.m4_data import encode_m4_round
from dcl.m4_models import make_m4_model

from train_m4_seed import DEVICE, generate_examples, evaluate
from train_m6_dagger import dagger_train_step


SEED = 1234
GRAPHS = 200
EPOCHS = 3
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

out = Path("data/results")

cp = torch.load(
    out / f"m4b_seed{SEED}_n40.pt",
    map_location=DEVICE,
    weights_only=True,
)

model = make_m4_model(
    cp["arch"],
    hidden_dim=cp["hidden_dim"],
    message_dim=cp["message_dim"],
).to(DEVICE)

model.load_state_dict(cp["model_state_dict"])

print("Generating original dataset...", flush=True)

original = generate_examples(
    800, 100_000, 200_000
)

validation = generate_examples(
    200, 300_000, 400_000
)

before = evaluate(model, validation)
print("BEFORE:", before, flush=True)

# Same fresh graphs and random-tape seeds
# as the DAgger iteration, but Luby drives
# the trajectories instead of the GNN.

fresh_oracle = []

print("Collecting fresh oracle examples...", flush=True)

for i in range(GRAPHS):
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=5_100_000 + i,
    )

    run = run_oracle(
        G,
        seed=6_100_000 + i,
        num_colors=4,
        max_rounds=64,
        save_trace=True,
    )

    fresh_oracle.extend(
        encode_m4_round(G, trace)
        for trace in run.trace
    )

print(
    "Fresh oracle examples:",
    len(fresh_oracle),
    flush=True,
)

assert fresh_oracle

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

rng = random.Random(SEED + 1)

updates = (
    len(original) + BATCH_SIZE - 1
) // BATCH_SIZE

for epoch in range(1, EPOCHS + 1):
    losses = []

    for _ in range(updates):
        loss = dagger_train_step(
            model,
            optimizer,
            original,
            fresh_oracle,
            batch_size=BATCH_SIZE,
            rng=rng,
        )
        losses.append(loss)

    print(
        f"epoch={epoch} "
        f"loss={np.mean(losses):.6f}",
        flush=True,
    )

after = evaluate(model, validation)

print("AFTER:", after, flush=True)

destination = (
    out / "m4b_seed1234_extra_oracle_control.pt"
)

torch.save(
    {
        **cp,
        "model_state_dict": model.state_dict(),
        "source": "extra_oracle_control",
        "iteration": 1,
        "extra_graphs": GRAPHS,
        "extra_examples": len(fresh_oracle),
        "val_metrics_before": before,
        "val_metrics": after,
    },
    destination,
)

print("Saved:", destination)
print("ORACLE CONTROL: PASS")
