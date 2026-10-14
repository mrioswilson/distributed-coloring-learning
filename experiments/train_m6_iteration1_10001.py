import random
from pathlib import Path

import numpy as np
import torch

from dcl.dagger import collect_dagger_rollout
from dcl.graphs import connected_random_regular_graph
from dcl.m4_models import make_m4_model
from dcl.oracle import run_oracle
from dcl.m4_data import encode_m4_round

from train_m4_seed import (
    DEVICE, generate_examples, evaluate
)
from train_m6_dagger import dagger_train_step


SEED = 10001
GRAPHS = 200
EPOCHS = 3
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

out = Path("data/results")
source = out / f"m4b_seed{SEED}_n40.pt"

cp = torch.load(
    source, map_location=DEVICE, weights_only=True
)

model = make_m4_model(
    "B",
    hidden_dim=cp["hidden_dim"],
    message_dim=cp["message_dim"],
).to(DEVICE)

model.load_state_dict(cp["model_state_dict"])

print("Generating original dataset...", flush=True)
original = generate_examples(800, 100_000, 200_000)
validation = generate_examples(200, 300_000, 400_000)

before = evaluate(model, validation)
print("BEFORE:", before, flush=True)

dagger = []
finished = blocked = disagreements = 0
off_oracle_rounds = 0

print("Collecting DAgger examples...", flush=True)

for i in range(GRAPHS):
    G = connected_random_regular_graph(
        n=40, degree=3, seed=5_100_000 + i
    )
    tape_seed = 6_100_000 + i

    result = collect_dagger_rollout(
        G, model,
        seed=tape_seed,
        max_rounds=64,
        device=DEVICE,
    )

    # Reference trajectory with exactly the same tape.
    reference = run_oracle(
        G, seed=tape_seed,
        num_colors=4,
        max_rounds=64,
        save_trace=True,
    )

    for t, example in enumerate(result.examples):
        if t >= len(reference.trace):
            off_oracle_rounds += 1
        else:
            oracle_example = encode_m4_round(
                G, reference.trace[t]
            )
            if not np.array_equal(
                example.x[:, :5],
                oracle_example.x[:, :5],
            ):
                off_oracle_rounds += 1

    dagger.extend(result.examples)

    finished += int(result.finished)
    blocked += (
        result.blocked_fixed
        + result.blocked_simultaneous
    )
    disagreements += result.action_disagreements

    if (i + 1) % 50 == 0:
        print(
            f"graphs={i+1}/{GRAPHS} "
            f"examples={len(dagger)} "
            f"disagreements={disagreements} "
            f"off_oracle_rounds={off_oracle_rounds}",
            flush=True,
        )

assert dagger, "No DAgger examples"

print("Finished:", finished, "/", GRAPHS)
print("Blocked:", blocked)
print("Disagreements:", disagreements)
print("Off-oracle rounds:", off_oracle_rounds)

# Persist the collected training examples.
np.savez_compressed(
    out / "m6b_seed10001_iter1_data.npz",
    x=np.stack([e.x for e in dagger]),
    edge_index=np.stack([e.edge_index for e in dagger]),
    yp=np.stack([e.y_participate for e in dagger]),
    yc=np.stack([e.y_candidate_offer for e in dagger]),
    ycommit=np.stack([e.y_commit for e in dagger]),
    mask=np.stack([e.decision_mask for e in dagger]),
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4
)

rng = random.Random(SEED + 1)
updates = (len(original) + BATCH_SIZE - 1) // BATCH_SIZE

print("Fine-tuning...", flush=True)

for epoch in range(1, EPOCHS + 1):
    losses = []

    for _ in range(updates):
        loss = dagger_train_step(
            model, optimizer,
            original, dagger,
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

torch.save(
    {
        **cp,
        "model_state_dict": model.state_dict(),
        "iteration": 1,
        "source": "dagger",
        "dagger_graphs": GRAPHS,
        "dagger_examples": len(dagger),
        "off_oracle_rounds": off_oracle_rounds,
        "val_metrics_before": before,
        "val_metrics": after,
    },
    out / "m6b_seed10001_iter1.pt",
)

print("M6 ITERATION 1: PASS")
