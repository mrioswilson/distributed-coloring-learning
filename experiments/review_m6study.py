"""Review validation and pin the budget extensions before opening final tests."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from statistics import mean

from dcl.study.config import Config
from dcl.study.storage import atomic_json, file_hash, now


def checkpoint_score(record):
    rollout, local = record["validation_rollouts"], record["validation_local"]
    return (rollout["success_rate"], -rollout["unsafe_per_state"], -rollout["mean_rounds"],
            local["action_correct_teacher_rate"], -local["loss"])


def collect(root, configs):
    with sqlite3.connect(f"file:{root / 'registry.sqlite'}?mode=ro", uri=True) as database:
        status = dict(database.execute("SELECT id,status FROM runs"))
    records = []
    for config in configs:
        directory = root / "runs" / config.id
        if status.get(config.id) != "complete":
            raise RuntimeError(f"Incomplete training: {config.id}")
        if Config.from_dict(json.loads((directory / "config.json").read_text())) != config:
            raise RuntimeError(f"Persisted configuration differs: {config.id}")
        history = json.loads((directory / "history.json").read_text())
        if [row["epoch"] for row in history] != list(range(1, config.epochs + 1)):
            raise RuntimeError(f"Incomplete epoch history: {config.id}")
        for filename in ["best.pt", "last.pt"] + [f"epoch-{epoch:03d}.pt" for epoch in config.checkpoint_epochs]:
            if not (directory / filename).is_file():
                raise RuntimeError(f"Missing {filename}: {config.id}")
        selected = max((row for row in history if row["validation_rollouts"]), key=checkpoint_score)
        rollout, local = selected["validation_rollouts"], selected["validation_local"]
        records.append({"id": config.id, "config": config.to_dict(),
                        "history_sha256": file_hash(directory / "history.json"),
                        "selected_epoch": selected["epoch"], "selected_score": list(checkpoint_score(selected)),
                        "successes": rollout["successes"], "graphs": rollout["graphs"],
                        "success_rate": rollout["success_rate"], "unsafe_commits": rollout["unsafe_commits"],
                        "unsafe_per_state": rollout["unsafe_per_state"], "mean_rounds": rollout["mean_rounds"],
                        "action_accuracy": local["action_correct_teacher_rate"], "local_loss": local["loss"],
                        "training_seconds": history[-1]["elapsed_seconds"] + history[-1]["data_generation_seconds"],
                        "last_successes": history[-1]["validation_rollouts"]["successes"],
                        "last_unsafe_commits": history[-1]["validation_rollouts"]["unsafe_commits"]})
    return records


def select_extensions(records):
    groups = defaultdict(list)
    for row in records:
        config = Config.from_dict(row["config"])
        if config.degree == 3 and config.seed < 5 and config.method == "COUNTERFACTUAL" and config.family == "REGULAR":
            groups[(config.graphs, config.tapes, config.updates_per_epoch)].append(row)
    summaries = []
    for (graphs, tapes, updates), rows in sorted(groups.items(), key=str):
        if {row["config"]["seed"] for row in rows} != set(range(5)) or len(rows) != 5:
            raise RuntimeError("Budget selection requires exactly the five paired discovery seeds")
        # Use the same priority as checkpoint selection, averaged over seeds.
        score = [mean(row["selected_score"][i] for row in rows) for i in range(5)]
        summaries.append({"graphs": graphs, "tapes": tapes, "updates_per_epoch": updates,
                          "score": score, "training_seconds_mean": mean(row["training_seconds"] for row in rows),
                          "discovery_ids": [row["id"] for row in rows]})
    pools = {"epochs": [row for row in summaries if row["updates_per_epoch"] is None],
             "updates": [row for row in summaries if row["updates_per_epoch"] == 256],
             "2400_trajectories": [row for row in summaries if row["updates_per_epoch"] is None
                                    and row["graphs"] * row["tapes"] == 2400]}
    if [len(pools[key]) for key in pools] != [9, 9, 5]:
        raise RuntimeError("Discovery budget grid is incomplete")
    winners = {key: max(rows, key=lambda row: (row["score"], -row["graphs"], -row["tapes"]))
               for key, rows in pools.items()}
    configs = {}
    for winner in winners.values():
        for degree in (3, 5, 8, 10):
            for seed in range(20):
                config = replace(Config(degree=degree, seed=seed), graphs=winner["graphs"], tapes=winner["tapes"],
                                 updates_per_epoch=winner["updates_per_epoch"])
                configs[config.id] = config
    return summaries, winners, list(configs.values())


def verify_main_pairs(root, configs):
    import torch
    available = {config.id: config for config in configs}
    pairs = []
    for degree in (3, 5, 8, 10):
        for seed in range(20):
            treatment = Config(degree=degree, seed=seed)
            control = replace(treatment, method="OWN_ONLY")
            if treatment.id not in available or control.id not in available:
                raise RuntimeError("The principal paired cohort is incomplete")
            a = torch.load(root / "runs" / treatment.id / "last.pt", map_location="cpu", weights_only=False)
            b = torch.load(root / "runs" / control.id / "last.pt", map_location="cpu", weights_only=False)
            for key in ("data_hash", "initial_state_hash", "updates", "rounds_seen", "nodes_seen"):
                if a[key] != b[key]:
                    raise RuntimeError(f"Paired {key} differs: degree={degree}, seed={seed}")
            names = [key for key in a["model_state_dict"] if key.startswith(("message1.", "participation_head.", "candidate_head."))]
            if not names or not all(torch.equal(a["model_state_dict"][key], b["model_state_dict"][key]) for key in names):
                raise RuntimeError(f"Paired P/C weights differ: degree={degree}, seed={seed}")
            pairs.append({"degree": degree, "seed": seed, "shared_tensor_count": len(names),
                          "same_data_initialization_updates_exposures_and_PC": True})
    return pairs


def read_plan(path):
    value = json.loads(path.read_text())
    if value["protocol"] != "m6study-v1":
        raise ValueError("Unknown protocol")
    configs = [Config.from_dict(value) for value in value["experiments"]]
    if len({config.id for config in configs}) != len(configs):
        raise ValueError("Duplicate configurations")
    return configs


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise RuntimeError(f"Overwrite refused: {path}")
    else:
        atomic_json(path, value)


def prepare(repo, root, output):
    if any((root / "evaluations").glob("*/*/*.json")) or (root / "frozen-final-test.json").exists():
        raise RuntimeError("Budget extensions must be selected before any final testing")
    with sqlite3.connect(f"file:{root / 'registry.sqlite'}?mode=ro", uri=True) as database:
        configs = [Config.from_dict(json.loads(row[0])) for row in database.execute("SELECT config FROM runs ORDER BY rowid")]
    records = collect(root, configs)
    summaries, winners, extensions = select_extensions(records)
    pairs = verify_main_pairs(root, configs)
    all_configs = {config.id: config for config in configs + extensions}
    plans = {}
    for name, cohort in (("phase-c-extension", extensions), ("phase-e-cohort", list(all_configs.values()))):
        path = repo / "configs/m6study" / f"{name}.json"
        write_once(path, {"protocol": "m6study-v1", "name": name, "experiments": [config.to_dict() for config in cohort]})
        plans[name] = {"path": str(path), "sha256": file_hash(path), "experiments": len(cohort)}
    value = {"created_utc": now(), "protocol": "m6study-v1", "final_test_observations_used": 0,
             "selection_rule": "Mean of the existing lexicographic checkpoint scores over discovery seeds 0..4; exact ties prefer fewer graphs, then tapes.",
             "selection_is_exploratory": True, "small_validation_is_not_final_test": True,
             "budget_summaries": summaries, "selected_extensions": winners, "plans": plans,
             "completed_at_review": len(configs), "additional_unique_training": len(all_configs) - len(configs),
             "paired_principal_runs": pairs, "records": records,
             "final_policies": {"all_models": ["best", "epoch-120"],
                                "principal_delta3_counterfactual_extra_epochs": [20, 40, 60, 80]},
             "review_complete": True}
    if output.exists():
        raise RuntimeError(f"Review already exists: {output}")
    atomic_json(output, value)
    print(json.dumps({"review": str(output), "winners": winners, "completed": len(configs),
                      "additional": value["additional_unique_training"], "planned_total": len(all_configs)}, indent=2))


def verify(root, plan, review_path):
    review = json.loads(review_path.read_text())
    if not review["review_complete"] or review["final_test_observations_used"] != 0:
        raise RuntimeError("A validation-only review is required")
    if file_hash(plan) != review["plans"]["phase-e-cohort"]["sha256"]:
        raise RuntimeError("The final cohort changed after validation review")
    records = collect(root, read_plan(plan))
    records_by_id = {row["id"]: row for row in records}
    for original in review["records"]:
        if records_by_id[original["id"]]["history_sha256"] != original["history_sha256"]:
            raise RuntimeError("A reviewed training history changed")
    pairs = verify_main_pairs(root, read_plan(plan))
    atomic_json(root / "reports/pre-final-verification.json", {"created_utc": now(), "completed": len(records),
                "review_sha256": file_hash(review_path), "paired_principal_runs": pairs, "records": records})
    print(f"Verified {len(records)} completed trainings and {len(pairs)} principal pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--root", default="data/m6study-release", type=Path)
    parser.add_argument("--review", default="data/m6study-release/reports/validation-review.json", type=Path)
    parser.add_argument("--plan", default="configs/m6study/phase-e-cohort.json", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        prepare(repo, args.root.resolve(), args.review.resolve())
    else:
        verify(args.root.resolve(), args.plan.resolve(), args.review.resolve())


if __name__ == "__main__":
    main()
