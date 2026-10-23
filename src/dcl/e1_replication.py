"""Canonical E1 retraining, freezing and audits, authorized 2026-10-21.

The original UniversalColorGNN and trainer are used unchanged. New checkpoint
bytes are authenticated by a separate frozen cohort index, never E1_INDEX.json.
"""
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from functools import partial
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
import torch

from dcl import e1_behavior as behavior
from dcl.study.config import digest
from dcl.study.storage import atomic_json, atomic_write, file_hash, now, resources
from dcl.universal import train as training
from dcl.universal.config import UConfig
from dcl.universal.evaluation import run_rollout, teacher_rollout
from dcl.universal.model import UniversalColorGNN, count_parameters

REPO = behavior.REPO
ROOT = REPO / "data/e1-retrained-20261021"
OUTPUT = REPO / "results/e1-retrained/2026-10-21"
PLAN = REPO / "configs/e1_behavior/retrained.json"
PROTOCOL = "e1-replication-v1"


def configs():
    base = UConfig.from_dict(json.loads((REPO / "configs/ucgnn/e1.json").read_text())["experiments"][0])
    return [replace(base, seed=s) for s in range(5)]


def prepare_training(root=ROOT):
    root = Path(root)
    # Authentication of every imported original source, even though its weights
    # are deliberately not available and not required for this reproduction.
    behavior.inspect_inputs()
    paths = [p for p in (REPO / "src/dcl").rglob("*.py")]
    source_hashes = {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)}
    plan = {"protocol": PROTOCOL, "upstream_commit": behavior.UPSTREAM,
            "cohort_origin": "newly-trained-replicas", "configs": [c.to_dict() for c in configs()],
            "canonical_config_sha256": file_hash(REPO / "configs/ucgnn/e1.json"),
            "sources": source_hashes, "comparison_plan_sha256": file_hash(PLAN),
            "device": "cpu", "threads_per_trainer": 1,
            "resource_guard": "same disk guard; CPU free-RAM floor 0.35GiB; original default is 2GiB",
            "checkpoint_selection": "epoch120 of all five replicas, fixed before test, no test selection"}
    plan = json.loads(json.dumps(plan))
    path = root / "training-plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise RuntimeError("Existing training plan/sources differ; overwrite refused")
    else:
        atomic_json(path, plan)
    for relative, expected in source_hashes.items():
        dest = root / "sources" / relative
        if dest.exists():
            if file_hash(dest) != expected:
                raise RuntimeError("Training source snapshot changed")
        else:
            atomic_write(dest, lambda f, r=relative: f.write((REPO / r).read_bytes()))
    return plan


def train_seeds(seeds, root=ROOT):
    root = Path(root)
    plan = prepare_training(root)
    canonical = configs()
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in range(5) for s in seeds):
        raise ValueError("Distinct original E1 seeds 0..4 required")
    # Resource preflight is process-local. Scientific code, gradients, random
    # streams, optimizer, datasets and validation remain the original trainer.
    prior_resources = training.resources
    training.resources = partial(resources, min_ram_gb=0.35)
    try:
        for seed in seeds:
            config = canonical[seed]
            print(f"START E1 NEW seed={seed} id={config.id} epochs={config.epochs}", flush=True)
            state = training.train(config, root, device="cpu", check_resources=True)
            if state != "complete":
                raise RuntimeError(f"E1 seed{seed} did not complete")
            if prepare_training(root) != plan:
                raise RuntimeError("Sources changed while training")
            print(f"COMPLETE E1 NEW seed={seed}", flush=True)
    finally:
        training.resources = prior_resources


def _train_logged(seed, root):
    path = Path(root) / "logs" / f"seed{seed}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        train_seeds([seed], root)
    return seed


def train_parallel(seeds, root=ROOT, workers=2):
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in range(5) for s in seeds):
        raise ValueError("Distinct E1 seeds 0..4 required before launching trainers")
    prepare_training(root)
    if workers not in (1, 2):
        raise ValueError("At most two CPU trainers fit the measured RAM budget")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        pending = [pool.submit(_train_logged, seed, str(root)) for seed in seeds]
        for completed in as_completed(pending):
            print(f"E1 new replica seed{completed.result()} completed", flush=True)


def freeze(root=ROOT):
    root = Path(root)
    plan = prepare_training(root)
    index = {"protocol": PROTOCOL, "cohort_origin": "newly-trained-replicas",
             "upstream_commit": behavior.UPSTREAM, "architecture": "UniversalColorGNN NORMALIZED",
             "method": "COUNTERFACTUAL", "parameters": 16035,
             "canonical_config_sha256": plan["canonical_config_sha256"], "sources": plan["sources"],
             "frozen_before_comparison": True, "checkpoints": []}
    for config in configs():
        path = root / "runs" / config.id / "epoch-120.pt"
        model, saved = training.load_model(path, "cpu")
        if type(model) is not UniversalColorGNN or UConfig.from_dict(saved["config"]) != config:
            raise RuntimeError("Replica architecture/configuration differs from canonical E1")
        if (saved["epoch"], saved["complete"], saved["parameters"], count_parameters(model)) != (120, True, 16035, 16035):
            raise RuntimeError("Replica is not the final complete E1 checkpoint")
        if saved["provenance"]["source_hashes"] != plan["sources"]:
            raise RuntimeError("Training provenance differs from frozen source snapshot")
        manifest_path = path.parent / "data_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if digest(manifest) != saved["data_hash"]:
            raise RuntimeError("Training/validation data manifest changed")
        if (len(manifest["train"]), len(manifest["validation"]), manifest["training_trajectories"]) != (800, 200, 2400):
            raise RuntimeError("Wrong E1 dataset budget")
        rows = saved["history"]
        expected_updates = 120 * ((manifest["training_rounds"] + 127) // 128)
        if len(rows) != 120 or saved["updates"] != expected_updates:
            raise RuntimeError("Wrong E1 optimization budget")
        index["checkpoints"].append({"seed": config.seed, "run": config.id,
                                     "checkpoint": str(path.relative_to(root)), "sha256": file_hash(path),
                                     "bytes": path.stat().st_size, "data_hash": saved["data_hash"],
                                     "data_manifest_sha256": file_hash(manifest_path), "updates": saved["updates"],
                                     "initial_state_hash": saved["initial_state_hash"]})
    path = root / "frozen.json"
    if path.exists():
        if json.loads(path.read_text()) != index:
            raise RuntimeError("Frozen cohort changed; overwrite refused")
    else:
        if (root / "comparison/manifest.json").exists():
            raise RuntimeError("Cannot freeze models after opening the comparison")
        atomic_json(path, index)
        atomic_json(root / "freeze-time.json", {"frozen_utc": now(), "index_sha256": file_hash(path)})
    return index


def compare(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output)
    freeze(root)
    behavior.run(PLAN, root / "comparison", root, root / "frozen.json")
    behavior.summarize(root / "comparison", output)


def audit(root=ROOT, output=OUTPUT):
    """Replay original scalar teacher/model runs and every saved local example."""
    root, output = Path(root), Path(output)
    models, inputs = behavior.load_models(root, root / "frozen.json")
    comparison = root / "comparison"
    manifest = json.loads((comparison / "manifest.json").read_text())
    if manifest["inputs"] != inputs:
        raise RuntimeError("Evaluation inputs changed")
    plan, teacher_checks, model_checks, example_checks, coloring_checks = manifest["plan"], 0, 0, 0, 0
    torch.set_num_threads(1)
    for case in manifest["graphs"]:
        path = comparison / "graphs" / (case["id"] + ".json")
        record = json.loads(path.read_text())
        if record["arrays_sha256"] != file_hash(path.with_suffix(".npz")):
            raise RuntimeError("Audit arrays changed")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as f:
            arrays = dict(f)
        n, edges = case["n"], arrays["edges"]
        for actor in ["luby_a", "luby_b", *(f"gnn_s{s}" for s in models)]:
            colors = arrays[actor + "_colors"]
            np.testing.assert_array_equal(behavior.terminal_status(colors, edges), arrays[actor + "_status"])
            coloring_checks += len(colors)
        for pool in ("a", "b"):
            for i in (0, len(arrays["tape_seeds_" + pool]) - 1):
                colors, rounds = teacher_rollout(n, edges, 4, int(arrays["tape_seeds_" + pool][i]), plan["max_rounds"])
                np.testing.assert_array_equal(colors, arrays[f"luby_{pool}_colors"][i])
                assert rounds == int(arrays[f"luby_{pool}_rounds"][i])
                teacher_checks += 1
        for seed, model in models.items():
            i = seed % plan["samples"]
            tape_seed = int(arrays["tape_seeds_b"][i])
            result, colors = run_rollout(model, n, edges, 4, tape_seed, plan["max_rounds"], diagnostics=False)
            np.testing.assert_array_equal(colors, arrays[f"gnn_s{seed}_colors"][i])
            assert result["rounds"] == int(arrays[f"gnn_s{seed}_rounds"][i])
            model_checks += 1
        for ex in record["examples"]:
            colors = np.asarray(ex["colors_before"], np.int64)
            rp, rc = np.asarray(ex["activation_draw"]), np.asarray(ex["candidate_draw"])
            _, p, c, m, _, _ = behavior.truth_from_draws(colors[None], rp[None], rc[None], edges)
            np.testing.assert_array_equal(p[0], ex["truth_p"])
            np.testing.assert_array_equal(c[0], ex["truth_c"])
            np.testing.assert_array_equal(m[0], ex["truth_commit"])
            model = models[ex["replica"]]
            ct, et, kt = torch.from_numpy(colors), torch.from_numpy(edges), torch.full((n,), 4)
            with torch.inference_mode():
                out = model(ct, torch.from_numpy(rp.astype(np.float32)), torch.from_numpy(rc.astype(np.float32)), et, kt)
                shared = model.commit_logits(ct, et, kt, torch.from_numpy(p[0]), torch.from_numpy(c[0])) > 0
            np.testing.assert_array_equal(out.participate.numpy(), ex["pred_p"])
            np.testing.assert_array_equal(out.candidate.numpy(), ex["pred_c"])
            np.testing.assert_array_equal((out.commit_logit > 0).numpy(), ex["pred_commit"])
            np.testing.assert_array_equal(shared.numpy(), ex["shared_commit"])
            example_checks += 1
    frozen = freeze(root)
    result = {"complete": True, "checked_utc": now(), "colorings_checked": coloring_checks,
              "teacher_scalar_replays": teacher_checks, "e1_scalar_replays": model_checks,
              "local_examples_replayed": example_checks, "checkpoints": len(frozen["checkpoints"]),
              "cohort_origin": "newly-trained-replicas", "independent_scientific_review": "pending"}
    atomic_json(output / "audit.json", result)
    return result
