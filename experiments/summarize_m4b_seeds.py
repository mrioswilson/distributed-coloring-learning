from pathlib import Path
import re

import pandas as pd


files = sorted(
    Path("data/results").glob(
        "m4b_seed*_extrapolation.csv"
    )
)

if not files:
    raise SystemExit(
        "No seed evaluations found."
    )

frames = []

for path in files:
    match = re.search(
        r"seed(\d+)",
        path.name,
    )

    if match is None:
        continue

    seed = int(match.group(1))

    df = pd.read_csv(path)
    df["seed"] = seed

    frames.append(df)

all_results = pd.concat(
    frames,
    ignore_index=True,
)

print()
print("SEEDS FOUND:")
print(
    sorted(
        all_results["seed"].unique()
    )
)

print()
print("N = 10240 BY SEED")
print()

last = (
    all_results[
        all_results["n"] == 10240
    ][
        [
            "seed",
            "success_rate",
            "open_action_accuracy",
            "false_commit",
            "missed_commit",
            "wrong_color",
            "mean_learned_rounds",
            "mean_oracle_rounds",
        ]
    ]
    .sort_values("seed")
)

print(
    last.to_string(
        index=False
    )
)

summary = (
    all_results
    .groupby("n")
    .agg(
        seeds=(
            "seed",
            "nunique",
        ),
        success_mean=(
            "success_rate",
            "mean",
        ),
        success_std=(
            "success_rate",
            "std",
        ),
        success_min=(
            "success_rate",
            "min",
        ),
        success_max=(
            "success_rate",
            "max",
        ),
        action_acc_mean=(
            "open_action_accuracy",
            "mean",
        ),
        action_acc_std=(
            "open_action_accuracy",
            "std",
        ),
        rounds_mean=(
            "mean_learned_rounds",
            "mean",
        ),
        oracle_rounds_mean=(
            "mean_oracle_rounds",
            "mean",
        ),
    )
    .reset_index()
)

print()
print("AGGREGATED RESULTS")
print()

print(
    summary.to_string(
        index=False
    )
)

summary.to_csv(
    "data/results/"
    "m4b_seed_summary.csv",
    index=False,
)

all_results.to_csv(
    "data/results/"
    "m4b_all_seeds.csv",
    index=False,
)

print()
print(
    "saved: "
    "data/results/m4b_seed_summary.csv"
)
print(
    "saved: "
    "data/results/m4b_all_seeds.csv"
)
