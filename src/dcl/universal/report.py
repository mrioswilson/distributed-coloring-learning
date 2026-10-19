"""Tables for the UCGNN study from the original per-graph JSON records.

Specialist references come from the published M6B final-test tables
(results/m6study/2026-10-17-final), restricted to the main configuration:
COUNTERFACTUAL, REGULAR, 800 graphs x 3 tapes, 120 epochs, fixed-epoch budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

SUM_KEYS = ["success", "finished", "rounds", "monochromatic_edges", "unsafe_commits", "fixed_neighbor_conflicts",
            "simultaneous_conflicts", "valid_commits_rejected", "states", "participation_correct", "candidate_correct",
            "commit_correct_own", "action_agree", "forbidden_offers", "availability_slot_errors", "availability_slots",
            "interval_deviation_nodes", "interval_deviation_sum", "uncolored_nodes", "teacher_identical_trajectory",
            "inference_seconds", "wall_seconds"]
GROUP = ["view", "experiment", "seed", "variant", "checkpoint_epoch", "degree", "seen_degree", "n", "distribution"]


def clopper_pearson(successes, trials, level=0.95):
    a = (1 - level) / 2
    low = 0.0 if successes == 0 else beta.ppf(a, successes, trials - successes + 1)
    high = 1.0 if successes == trials else beta.ppf(1 - a, successes + 1, trials - successes)
    return float(low), float(high)


def load_records(root, split):
    rows = []
    for path in sorted((Path(root) / "evaluations" / split).glob("*/*.json")):
        if path.name == "view.json":
            continue
        record = json.loads(path.read_text())
        by_k = record.pop("by_k")
        record["training_degrees"] = "+".join(map(str, record["training_degrees"]))
        record["_by_k"] = by_k
        rows.append(record)
    return rows


def regime(row):
    degrees = [int(d) for d in str(row["training_degrees"]).split("+")]
    if row["degree"] in degrees:
        return "seen"
    return "interpolation" if min(degrees) < row["degree"] < max(degrees) else "extrapolation"


def summarize(records, keys=GROUP):
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "_by_k"} for r in records])
    frame["regime"] = frame.apply(regime, axis=1)
    frame["teacher_rounds"] = frame.get("teacher_rounds", np.nan)
    table = frame.groupby(keys + ["regime"], dropna=False).agg(
        graphs=("graph_id", "count"), **{k: (k, "sum") for k in SUM_KEYS if k in frame},
        teacher_rounds=("teacher_rounds", "sum"), gpu_peak_mb=("gpu_peak_bytes", lambda x: x.max() / 2**20),
        inference_seconds_max=("inference_seconds", "max")).reset_index()
    table["success_rate"] = table.success / table.graphs
    bounds = [clopper_pearson(int(s), int(n)) for s, n in zip(table.success, table.graphs)]
    table["exact_low"], table["exact_high"] = zip(*bounds)
    table["mean_rounds"] = table.rounds / table.graphs
    table["mean_teacher_rounds"] = table.teacher_rounds / table.graphs
    for name, numerator in (("participation_accuracy", "participation_correct"), ("candidate_accuracy", "candidate_correct"),
                            ("commit_accuracy", "commit_correct_own"), ("action_agreement", "action_agree"),
                            ("interval_deviation_rate", "interval_deviation_nodes"), ("unsafe_per_state", "unsafe_commits"),
                            ("forbidden_offer_rate", "forbidden_offers")):
        table[name] = table[numerator] / table.states
    table["availability_error_rate"] = table.availability_slot_errors / table.availability_slots
    table["mean_inference_seconds"] = table.inference_seconds / table.graphs
    return table


def by_k(records, keys=("view", "experiment", "seed", "degree", "n")):
    rows = []
    for r in records:
        for k, counts in r["_by_k"].items():
            rows.append({**{key: r[key] for key in keys}, "k": int(k), **counts})
    frame = pd.DataFrame(rows)
    table = frame.groupby(list(keys) + ["k"]).sum(numeric_only=True).reset_index()
    table["candidate_accuracy"] = table.candidate_correct / table.states
    table["commit_accuracy"] = table.commit_correct / table.states
    table["unsafe_rate"] = table.unsafe / table.states
    table["interval_deviation_rate"] = table.interval_deviation_nodes / table.states
    return table


def specialists(results_dir):
    """Main M6B Counterfactual specialists, pooled over replicas, by degree/size/family/policy."""
    frame = pd.read_csv(Path(results_dir) / "success-by-seed.csv")
    main = frame[(frame.method == "COUNTERFACTUAL") & (frame.training_family == "REGULAR") & (frame.training_graphs == 800)
                 & (frame.training_tapes == 3) & (frame.updates_per_epoch == 0) & (frame.training_epochs == 120)]
    main = main[main.checkpoint_policy.isin(["epoch-120", "best"])]
    grouped = main.groupby(["checkpoint_policy", "degree", "n", "distribution"]).agg(
        replicas=("seed", "nunique"), evaluations=("graphs", "sum"), successes=("successes", "sum"),
        min_replica_rate=("success_rate", "min"), rounds_sum=("rounds_sum", "sum"),
        unsafe_commits=("unsafe_commits", "sum"), states=("states", "sum")).reset_index()
    grouped["success_rate"] = grouped.successes / grouped.evaluations
    grouped["mean_rounds"] = grouped.rounds_sum / grouped.evaluations
    return grouped


def specialist_by_k(results_dir):
    frame = pd.read_csv(Path(results_dir) / "k-by-seed.csv")
    main = frame[(frame.method == "COUNTERFACTUAL") & (frame.training_family == "REGULAR") & (frame.training_graphs == 800)
                 & (frame.training_tapes == 3) & (frame.updates_per_epoch == 0) & (frame.training_epochs == 120)
                 & (frame.checkpoint_policy == "epoch-120")]
    table = main.groupby(["degree", "n", "k"]).sum(numeric_only=True).reset_index()
    table["candidate_accuracy"] = table.candidate_correct / table.states.where(table.states > 0)
    return table[["degree", "n", "k", "states", "candidate_correct", "candidate_accuracy", "unsafe", "forbidden"]]


def write_tables(root, split, output, m6_results=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(root, split)
    if not records:
        raise RuntimeError(f"No evaluation records for split {split}")
    summary = summarize(records)
    summary.to_csv(output / f"{split}-summary.csv", index=False)
    overall = summarize(records, keys=["view", "experiment", "seed", "variant", "checkpoint_epoch", "degree", "seen_degree", "n"])
    overall.to_csv(output / f"{split}-by-size.csv", index=False)
    by_k(records).to_csv(output / f"{split}-by-k.csv", index=False)
    if m6_results is not None:
        specialists(m6_results).to_csv(output / "m6b-specialists-final-test.csv", index=False)
        specialist_by_k(m6_results).to_csv(output / "m6b-specialists-by-k.csv", index=False)
    return summary, overall
