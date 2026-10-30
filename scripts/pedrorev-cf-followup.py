#!/usr/bin/env python3
"""Experiment F (protocol section 8): neighbor-message interventions for BE.

Stages: train -> freeze -> battery -> respond -> functional. Battery v2 is a
new namespace generated only after the follow-up freeze. Run from the
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
from dcl.pedrorev.banks import neighbor_bank
from dcl.pedrorev.evaluation import queries, responses, state_bank
from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.protocol import passed_run, summarize_runs
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash

CONFIG = Path("configs/pedrorev-cf-followup.json")
DATA = Path("data/pedrorev-cf")
FDATA = Path("data/pedrorev-cf-followup")
OUT = Path("results/pedro-revision/cf-followup")
TEACHER = "BE_BASE_D3"


def config():
    return json.loads(CONFIG.read_text())


def _train(job):
    cfg, seed = job
    torch.set_num_threads(1)
    directory, _ = neighbor_bank(cfg, TEACHER, seed, cfg["low_total_labels"][TEACHER], DATA)
    return train(cfg, TEACHER, "normalized", seed, directory, DATA / "runs" / TEACHER.lower() / f"s{seed}" / "cf-neighbor",
                 "counterfactual", cfg["updates"], cfg["checkpoints"], development=False)


def stage_train(cfg):
    with ProcessPoolExecutor(max_workers=5) as pool:
        list(pool.map(_train, [(cfg, s) for s in cfg["seeds"]]))


def stage_freeze(cfg):
    if (FDATA / "evaluation-banks").exists() or (OUT / "freeze.json").exists():
        raise RuntimeError("Freeze must precede battery v2 and happen once")
    original = json.loads(Path("results/pedro-revision/cf-experiment/freeze.json").read_text())["checkpoints"]
    rows = [dict(r) for r in original if r["teacher"] == TEACHER and r["update"] == 4000
            and r["arm"] in cfg["followup_reference_arms"]]
    for seed in cfg["seeds"]:
        d = DATA / "runs" / TEACHER.lower() / f"s{seed}" / "cf-neighbor"
        rec = json.loads((d / "u004000-record.json").read_text())
        if file_hash(d / "u004000.pt") != rec["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint hash differs")
        rows.append(dict(teacher=TEACHER, seed=seed, arm="cf-neighbor", update=4000, checkpoint=str(d / "u004000.pt"),
                         sha256=rec["checkpoint_sha256"], initial_hash=rec["initial_hash"],
                         native_labels=rec["native_labels"], counterfactual_labels=rec["counterfactual_labels"],
                         total_acquired_component_labels=rec["total_acquired_component_labels"],
                         data_sha256=rec["identity"]["data_sha256"], training_seconds=rec["training_seconds"]))
    atomic_json(OUT / "freeze.json", dict(config=cfg, config_sha256=file_hash(CONFIG), sources=source_hashes(),
        checkpoints=rows, frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


def stage_battery(cfg):
    bank, manifest = state_bank(cfg, TEACHER, FDATA)
    _, qs = queries(cfg, TEACHER, FDATA)
    atomic_json(OUT / "battery.json", dict(manifest, queries=int(len(qs["y"])), positives=int(qs["y"].sum()),
        query_sha256=file_hash(FDATA / "evaluation-banks" / TEACHER.lower() / "queries.npz")))


def _respond(job):
    cfg, row = job
    torch.set_num_threads(1)
    target = FDATA / "responses" / f"s{row['seed']}-{row['arm']}.npy"
    if not target.exists():
        if file_hash(row["checkpoint"]) != row["sha256"]:
            raise RuntimeError("Frozen checkpoint changed")
        bank, _ = state_bank(cfg, TEACHER, FDATA)
        _, qs = queries(cfg, TEACHER, FDATA)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, responses(TEACHER, row["checkpoint"], row["seed"], bank, qs))
    return dict(path=str(target), sha256=file_hash(target))


def _functional(job):
    cfg, row = job
    torch.set_num_threads(1)
    target = OUT / "functional" / f"s{row['seed']}-{row['arm']}.json"
    if target.exists():
        return
    fc = dict(cfg, namespace=cfg["functional_namespace"], final_runs=cfg["functional_runs"], final_sizes=cfg["functional_sizes"])
    suite = cases(fc, "final", 0)
    model = initialize(TEACHER, "normalized", row["seed"])
    model.load_state_dict(torch.load(row["checkpoint"], weights_only=False)["model"])
    rows, seconds = evaluate(model, TEACHER, suite)
    family = {m["case_id"]: m["family"] for _, m in suite}
    by = {}
    for r in rows:
        if not passed_run(r):
            by[family[r["case_id"]]] = by.get(family[r["case_id"]], 0) + 1
    atomic_json(target, dict(seed=row["seed"], arm=row["arm"], checkpoint_sha256=row["sha256"], seconds=seconds,
                             failures_by_family=by, **summarize_runs(rows, [m["case_id"] for _, m in suite])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "freeze", "battery", "respond", "functional"])
    args = parser.parse_args()
    cfg = config()
    torch.set_num_threads(1)
    if args.stage == "train":
        stage_train(cfg)
    elif args.stage == "freeze":
        stage_freeze(cfg)
    elif args.stage == "battery":
        stage_battery(cfg)
    else:
        rows = json.loads((OUT / "freeze.json").read_text())["checkpoints"]
        fn = _respond if args.stage == "respond" else _functional
        with ProcessPoolExecutor(max_workers=10) as pool:
            out = list(pool.map(fn, [(cfg, r) for r in rows]))
        if args.stage == "respond":
            atomic_json(OUT / "responses-index.json", out)


if __name__ == "__main__":
    main()
