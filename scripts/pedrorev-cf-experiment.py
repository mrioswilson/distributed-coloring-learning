#!/usr/bin/env python3
"""Experiments A/B/C of the coauthor-review revision (protocol pedrorev-cf-v1).

Stages: banks -> train -> freeze -> battery -> respond -> functional.
Analysis is separate (scripts/pedrorev-cf-analysis.py). Run from the
repository root with PYTHONPATH=src.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from dcl.multicf.protocol import cases, source_hashes
from dcl.multicf.training import train
from dcl.pedrorev.banks import acquire_native, holdout_bank, truncated_cf_bank
from dcl.pedrorev.evaluation import queries, responses, state_bank
from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.protocol import passed_run, summarize_runs
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash

CONFIG = Path("configs/pedrorev-cf.json")
DATA = Path("data/pedrorev-cf")
OUT = Path("results/pedro-revision/cf-experiment")
METHOD = {"passive": "passive", "matched": "matched", "cf": "counterfactual"}


def load_config():
    return json.loads(CONFIG.read_text())


def arms(config, teacher):
    base = ["passive-low", "matched-low", "cf-high", "passive-high", "matched-high", "cf-low"]
    return base + (["cf-noforce", "cf-lateocc"] if teacher == "BE_BASE_D3" else [])


def bank_for(config, teacher, seed, arm):
    low = config["low_total_labels"][teacher]
    if arm in ("passive-low", "matched-low", "cf-high"):
        return acquire_native(config, teacher, seed, low, DATA)
    if arm in ("passive-high", "matched-high"):
        _, base = acquire_native(config, teacher, seed, low, DATA)
        return acquire_native(config, teacher, seed, base["native_labels"] + base["counterfactual_labels"], DATA)
    if arm == "cf-low":
        return truncated_cf_bank(config, teacher, seed, low, DATA)
    if arm == "cf-noforce":
        return holdout_bank(config, teacher, seed, low, "noforce", DATA)
    if arm == "cf-lateocc":
        return holdout_bank(config, teacher, seed, low, "late-occupied", DATA)
    raise ValueError(arm)


def run_dir(teacher, seed, arm):
    return DATA / "runs" / teacher.lower() / f"s{seed}" / arm


def banks(config):
    summary = []
    for teacher in config["teachers"]:
        for seed in config["seeds"]:
            for arm in arms(config, teacher):
                directory, manifest = bank_for(config, teacher, seed, arm)
                summary.append(dict(teacher=teacher, seed=seed, arm=arm, bank=str(directory),
                    sha256=manifest["sha256"], native_labels=manifest["native_labels"],
                    counterfactual_labels=manifest["counterfactual_labels"] if arm.startswith("cf") else 0,
                    stored_counterfactual_labels=manifest["counterfactual_labels"],
                    total_charged=manifest["native_labels"] + (manifest["counterfactual_labels"] if arm.startswith("cf") else 0),
                    modification=manifest.get("modification")))
                print(json.dumps(summary[-1]), flush=True)
    atomic_json(OUT / "banks.json", summary)


def _train(job):
    config, teacher, seed, arm = job
    torch.set_num_threads(1)
    directory, _ = bank_for(config, teacher, seed, arm)
    method = METHOD[arm.split("-")[0]]
    return train(config, teacher, config["architecture"], seed, directory, run_dir(teacher, seed, arm), method,
                 config["updates"], config["checkpoints"], development=False)


def training(config, teachers=None):
    jobs = [(config, t, s, a) for t in (teachers or config["teachers"]) for s in config["seeds"] for a in arms(config, t)]
    tick = time.time()
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        results = list(pool.map(_train, jobs))
    print(json.dumps(dict(trained=len(results), seconds=round(time.time() - tick))), flush=True)


def freeze(config):
    """Hash every checkpoint before any battery exists."""
    if (DATA / "evaluation-banks").exists():
        raise RuntimeError("A battery already exists; freeze must precede it")
    rows = []
    for teacher in config["teachers"]:
        for seed in config["seeds"]:
            for arm in arms(config, teacher):
                for u in config["checkpoints"]:
                    record = json.loads((run_dir(teacher, seed, arm) / f"u{u:06d}-record.json").read_text())
                    path = run_dir(teacher, seed, arm) / f"u{u:06d}.pt"
                    if file_hash(path) != record["checkpoint_sha256"]:
                        raise RuntimeError("Checkpoint hash differs")
                    rows.append(dict(teacher=teacher, seed=seed, arm=arm, update=u, checkpoint=str(path),
                        sha256=record["checkpoint_sha256"], initial_hash=record["initial_hash"],
                        native_labels=record["native_labels"], counterfactual_labels=record["counterfactual_labels"],
                        total_acquired_component_labels=record["total_acquired_component_labels"],
                        native_examples_used=record["native_examples_used"],
                        counterfactual_labels_used=record["counterfactual_labels_used"],
                        presentations=record["component_presentations"], training_seconds=record["training_seconds"],
                        data_sha256=record["identity"]["data_sha256"]))
    target = OUT / "freeze.json"
    if target.exists():
        raise RuntimeError("Freeze already written")
    atomic_json(target, dict(config=config, config_sha256=file_hash(CONFIG), sources=source_hashes(),
        protocol_doc_sha256=file_hash("docs/pedro-revision/PROTOCOL_PEDROREV_CF.md"), checkpoints=rows,
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    print(json.dumps(dict(frozen=len(rows))), flush=True)


def battery(config):
    for teacher in config["teachers"]:
        bank, manifest = state_bank(config, teacher, DATA)
        _, qs = queries(config, teacher, DATA)
        atomic_json(OUT / f"battery-{teacher.lower()}.json", dict(manifest, queries=int(len(qs["y"])),
            positives=int(qs["y"].sum()), query_sha256=file_hash(DATA / "evaluation-banks" / teacher.lower() / "queries.npz")))
        print(teacher, manifest["roots"], len(qs["y"]), flush=True)


def _respond(job):
    config, row = job
    torch.set_num_threads(1)
    target = DATA / "responses" / row["teacher"].lower() / f"s{row['seed']}-{row['arm']}-u{row['update']:06d}.npy"
    if target.exists():
        return str(target)
    if file_hash(row["checkpoint"]) != row["sha256"]:
        raise RuntimeError("Frozen checkpoint changed")
    bank, _ = state_bank(config, row["teacher"], DATA)
    _, qs = queries(config, row["teacher"], DATA)
    logits = responses(row["teacher"], row["checkpoint"], row["seed"], bank, qs)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, logits)
    return str(target)


def respond(config):
    frozen = json.loads((OUT / "freeze.json").read_text())["checkpoints"]
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        done = list(pool.map(_respond, [(config, r) for r in frozen]))
    atomic_json(OUT / "responses-index.json", [dict(path=p, sha256=file_hash(p)) for p in done])


def _functional(job):
    config, row = job
    torch.set_num_threads(1)
    target = OUT / "functional" / row["teacher"].lower() / f"s{row['seed']}-{row['arm']}.json"
    if target.exists():
        return json.loads(target.read_text())
    if file_hash(row["checkpoint"]) != row["sha256"]:
        raise RuntimeError("Frozen checkpoint changed")
    cfg = dict(config, namespace=config["functional_namespace"], final_runs=config["functional_runs"],
               final_sizes=config["functional_sizes"])
    suite = cases(cfg, "final", 0)
    model = initialize(row["teacher"], config["architecture"], row["seed"])
    model.load_state_dict(torch.load(row["checkpoint"], weights_only=False)["model"])
    rows, seconds = evaluate(model, row["teacher"], suite)
    summary = dict(teacher=row["teacher"], seed=row["seed"], arm=row["arm"], update=row["update"],
                   checkpoint_sha256=row["sha256"], seconds=seconds,
                   **summarize_runs(rows, [m["case_id"] for _, m in suite]),
                   failures_by_family={})
    family = {m["case_id"]: m["family"] for _, m in suite}
    for r in rows:
        if not passed_run(r):
            f = family[r["case_id"]]
            summary["failures_by_family"][f] = summary["failures_by_family"].get(f, 0) + 1
    atomic_json(target, summary)
    return summary


def functional(config):
    frozen = [r for r in json.loads((OUT / "freeze.json").read_text())["checkpoints"] if r["update"] == config["primary_update"]]
    cfg = dict(config, namespace=config["functional_namespace"], final_runs=config["functional_runs"],
               final_sizes=config["functional_sizes"])
    for teacher in config["teachers"]:
        path = OUT / "functional" / f"exact-{teacher.lower()}.json"
        if not path.exists():
            rows, _ = evaluate(None, teacher, cases(cfg, "final", 0))
            s = summarize_runs(rows, [m["case_id"] for _, m in cases(cfg, "final", 0)])
            if not s["pass_all"]:
                raise RuntimeError("Exact teacher failed on functional battery")
            atomic_json(path, s)
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        list(pool.map(_functional, [(config, r) for r in frozen]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["banks", "train", "freeze", "battery", "respond", "functional"])
    parser.add_argument("--teachers", nargs="+")
    args = parser.parse_args()
    config = load_config()
    torch.set_num_threads(1)
    if args.stage == "banks":
        banks(config)
    elif args.stage == "train":
        training(config, args.teachers)
    elif args.stage == "freeze":
        freeze(config)
    elif args.stage == "battery":
        battery(config)
    elif args.stage == "respond":
        respond(config)
    else:
        functional(config)


if __name__ == "__main__":
    main()
