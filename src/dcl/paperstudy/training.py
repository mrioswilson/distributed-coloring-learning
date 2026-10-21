"""Deterministic paired supervised runs with exact epoch-boundary recovery."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from dcl.study.config import digest, seed_for
from dcl.study.storage import atomic_json, file_hash, provenance, save_checkpoint
from dcl.universal.config import model_seed
from dcl.universal.data import collect_examples, make_batch
from dcl.universal.model import count_parameters
from dcl.universal.train import restore_rng, rng_state, training_step, validate_local, validate_rollouts
from .config import ROOT, Run
from .models import make_model


def weights_hash(state):
    result = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        result.update(key.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def training_sources():
    paths = [*Path("src/dcl/universal").glob("*.py"), *Path("src/dcl/study").glob("*.py"),
             *[Path("src/dcl/paperstudy") / f for f in ("models.py", "training.py", "config.py", "teacher.py")],
             *[Path("src/dcl") / f for f in ("m4_data.py", "oracle.py", "graphs.py")]]
    return {str(p): file_hash(p) for p in sorted(paths)}


def initialize(run, device):
    seed = model_seed(run.seed, "initialization")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = make_model(run.treatment).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run.training_config().lr)
    order = np.random.default_rng(model_seed(run.seed, "minibatches"))
    cf = np.random.default_rng(model_seed(run.seed, "counterfactual"))
    return model, optimizer, order, cf


def dataset(run, root):
    config = run.training_config()
    cyclic = run.treatment == "cyclic_teacher"
    path = root / "datasets" / f"s{run.seed}-{'cyclic' if cyclic else 'uniform'}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        return saved["train"], saved["validation"], saved["manifest"]
    train, records = collect_examples(config, "train")
    validation, val_records = collect_examples(config, "validation")
    if cyclic:
        from .teacher import teacher_trajectory
        def convert(old_records):
            from dcl.study.graphs import generate_graph
            from dcl.m4_data import _edge_index
            result = []
            for record in old_records:
                edges = _edge_index(generate_graph(record["n"], record["degree"], record["family"], record["graph_seed"]))
                for trajectory in record["trajectories"]:
                    examples = teacher_trajectory(record["n"], edges, record["degree"]+1, trajectory["tape_seed"], True)
                    result.extend(examples)
                    trajectory["rounds"] = len(examples)
                    trajectory["tape_hash"] = digest([trajectory["tape_seed"], "cyclic", len(examples)])
            return result
        train, validation = convert(records), convert(val_records)
    if {r["fingerprint"] for r in records} & {r["fingerprint"] for r in val_records}:
        raise RuntimeError("Train/validation graph overlap")
    manifest = {"train": records, "validation": val_records, "training_rounds": len(train),
                "validation_rounds": len(validation), "teacher": "cyclic" if cyclic else "uniform",
                "trajectories": config.graphs_per_degree[0] * config.tapes}
    save_checkpoint(path, {"train": train, "validation": validation, "manifest": manifest})
    return train, validation, manifest


def train(run, root=ROOT, device="cuda", stop_requested=lambda: False, stop_after_epoch=None):
    root = Path(root)
    run_dir = root / "runs" / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    configuration = run.to_dict()
    config_path = run_dir / "config.json"
    if config_path.exists() and digest(json.loads(config_path.read_text())) != digest(configuration):
        raise RuntimeError("Existing run has a different configuration")
    atomic_json(config_path, configuration)
    config = run.training_config()
    model, optimizer, order, cf = initialize(run, device)
    initial_hash = weights_hash(model.state_dict())
    train_examples, val_examples, manifest = dataset(run, root)
    atomic_json(run_dir / "data_manifest.json", manifest)
    source = training_sources()
    source_id, data_hash = digest(source), digest(manifest)
    history, updates, start_epoch, elapsed, best = [], 0, 1, 0., None
    metadata = provenance()
    if (run_dir / "last.pt").exists():
        saved = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if saved["paper_config"] != configuration or saved["training_sources_sha256"] != source_id:
            raise RuntimeError("Exact resume refused: configuration/training source changed")
        if saved["data_hash"] != data_hash or saved["provenance"]["versions"] != metadata["versions"] or saved["device"] != device:
            raise RuntimeError("Exact resume refused: dataset/runtime changed")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        restore_rng(saved["rng"], order, cf)
        history, updates, start_epoch = saved["history"], saved["updates"], saved["epoch"] + 1
        elapsed, best, metadata = saved["elapsed_seconds"], saved["best_score"], saved["provenance"]
        if saved["complete"]:
            return "complete"
    else:
        save_checkpoint(run_dir / "initial.pt", {"model_state_dict": model.state_dict(), "sha256": initial_hash})
    print(f"{run.id} rounds={len(train_examples)} parameters={count_parameters(model)} start={start_epoch}", flush=True)
    for epoch in range(start_epoch, config.epochs + 1):
        if stop_requested():
            return "paused"
        tick = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        losses, weighted = {}, 0
        permutation = order.permutation(len(train_examples))
        tape_start_hash = digest(cf.bit_generator.state)
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            batch = make_batch([train_examples[int(i)] for i in indices], device)
            metrics = training_step(model, optimizer, batch, config, epoch, cf)
            nodes = metrics.pop("decision_nodes")
            weighted += nodes
            for key, value in metrics.items():
                losses[key] = losses.get(key, 0.) + nodes * value
            updates += 1
        train_seconds = time.perf_counter() - tick
        local = validate_local(model, val_examples, config, device)
        rollouts, is_best = None, False
        if epoch == 1 or epoch % config.val_every == 0 or epoch == config.epochs:
            rollouts = validate_rollouts(model, config)
            score = [rollouts["success_rate"], -rollouts["unsafe_per_state"], -rollouts["mean_rounds"],
                     local["action_correct_teacher_rate"], -local["loss"]]
            if best is None or tuple(score) > tuple(best):
                best, is_best = score, True
        elapsed += time.perf_counter() - tick
        row = {"epoch": epoch, "updates": updates, "train": {k:v/weighted for k,v in losses.items()},
               "validation_local": local, "validation_rollouts": rollouts, "train_seconds": train_seconds,
               "elapsed_seconds": elapsed, "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
               "minibatch_order_hash": digest(permutation.tolist()), "counterfactual_rng_start_hash": tape_start_hash,
               "counterfactual_rng_end_hash": digest(cf.bit_generator.state)}
        history.append(row)
        checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch, "updates": updates, "config": config.to_dict(), "paper_config": configuration,
                      "rng": rng_state(order, cf), "history": history, "elapsed_seconds": elapsed, "best_score": best,
                      "data_hash": data_hash, "initial_state_hash": initial_hash, "parameters": count_parameters(model),
                      "training_sources": source, "training_sources_sha256": source_id, "provenance": metadata,
                      "complete": epoch == config.epochs, "device": device}
        if is_best:
            save_checkpoint(run_dir / "best.pt", checkpoint)
        save_checkpoint(run_dir / "last.pt", checkpoint)
        if epoch in config.checkpoint_epochs:
            p = run_dir / f"epoch-{epoch:03d}.pt"
            if p.exists():
                # Recovery after atomic epoch save but before last/history may
                # encounter the same weights only; never overwrite a different epoch.
                old = torch.load(p, map_location="cpu", weights_only=False)
                if weights_hash(old["model_state_dict"]) != weights_hash(model.state_dict()):
                    raise RuntimeError("Historical epoch checkpoint would be overwritten")
            else:
                save_checkpoint(p, checkpoint)
        atomic_json(run_dir / "history.json", history)
        atomic_json(run_dir / "progress.json", {"run": run.id, "epoch": epoch, "updates": updates,
                                                "elapsed_seconds": elapsed, "complete": checkpoint["complete"]})
        print(f"{run.id} epoch={epoch:03d} loss={row['train']['loss']:.5f} c={local['c_correct_rate']:.5f} "
              f"commit={local['commit_correct_own_rate']:.5f} seconds={elapsed:.1f}", flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch and not checkpoint["complete"]:
            return "paused"
    return "complete"


def load_model(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = make_model(saved["paper_config"]["treatment"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    return model.eval(), saved


def promotion(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    last = saved["history"][-1]["validation_local"]
    checks = {"completed": saved["complete"], "participation": last["p_correct_rate"] >= .98,
              "candidate": last["c_correct_rate"] >= .90, "commit": last["commit_correct_own_rate"] >= .95}
    return {"promoted": all(checks.values()), "checks": checks, "validation": last, "epoch": saved["epoch"]}
