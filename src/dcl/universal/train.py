"""Counterfactual supervised training of the Universal Color GNN.

L = L_PARTICIPATE + L_CANDIDATE + 0.5 L_COMMIT_OWN + 0.5 L_COMMIT_COUNTERFACTUAL.
Warmup epochs (<= config.warmup) train COMMIT on the teacher's proposals with
concordant teacher labels, as in M6B. COMMIT labels are always recomputed with
the exact rule for the proposals that the COMMIT stage actually receives.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

from dcl.study.config import digest, seed_for
from dcl.study.storage import atomic_json, provenance, resources, save_checkpoint
from .config import model_seed
from .data import collect_examples, make_batch
from .evaluation import build_case, cohort_cases, run_rollout
from .model import UniversalColorGNN, count_parameters
from .rules import exact_commit_labels, make_counterfactual_proposals


def initialize(config, device):
    seed = model_seed(config.seed, "initialization")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = UniversalColorGNN(config.hidden_dim, config.message_dim, config.variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    order = np.random.default_rng(model_seed(config.seed, "minibatches"))
    counterfactual = np.random.default_rng(model_seed(config.seed, "counterfactual"))
    return model, optimizer, order, counterfactual


def rng_state(order, counterfactual):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "order": copy.deepcopy(order.bit_generator.state), "counterfactual": copy.deepcopy(counterfactual.bit_generator.state)}


def restore_rng(value, order, counterfactual):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in value["cuda"]])
    order.bit_generator.state = value["order"]
    counterfactual.bit_generator.state = value["counterfactual"]


def _np(t):
    return t.detach().cpu().numpy()


def training_step(model, optimizer, batch, config, epoch, cf_rng):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(*batch.inputs())
    mask = batch.mask
    device = batch.colors.device
    p_loss = F.binary_cross_entropy_with_logits(out.participation_logit[mask], batch.y_participate[mask])
    c_loss = F.cross_entropy(out.candidate_logits[mask], batch.y_offer[mask])
    colors, edges, palette = _np(batch.colors), _np(batch.edge_index), _np(batch.palette)
    teacher_p, teacher_c = _np(batch.y_participate) > 0.5, _np(batch.y_offer)
    if epoch <= config.warmup:
        # Teacher proposals; their exact labels coincide with the teacher's commits.
        expert = model.commit_logits(batch.colors, batch.edge_index, batch.palette, batch.y_participate > 0.5, batch.y_offer)
        own_loss = cf_loss = F.binary_cross_entropy_with_logits(expert[mask], batch.y_commit[mask])
    else:
        own_p, own_c = _np(out.participate), _np(out.candidate)
        own_target = torch.from_numpy(exact_commit_labels(colors, edges, own_p, own_c).astype(np.float32)).to(device)
        own_loss = F.binary_cross_entropy_with_logits(out.commit_logit[mask], own_target[mask])
        if config.method == "COUNTERFACTUAL":
            cf_p, cf_c = make_counterfactual_proposals(colors, edges, palette, teacher_p, teacher_c, cf_rng)
            cf_target = torch.from_numpy(exact_commit_labels(colors, edges, cf_p, cf_c).astype(np.float32)).to(device)
        else:
            cf_p, cf_c, cf_target = own_p, own_c, own_target
        cf_logits = model.commit_logits(batch.colors, batch.edge_index, batch.palette,
                                        torch.from_numpy(cf_p).to(device), torch.from_numpy(cf_c).to(device))
        cf_loss = F.binary_cross_entropy_with_logits(cf_logits[mask], cf_target[mask])
    loss = p_loss + c_loss + 0.5 * own_loss + 0.5 * cf_loss
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    if missing:
        raise RuntimeError(f"Missing parameter gradients: {missing}")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite training loss")
    optimizer.step()
    return {"loss": float(loss.detach()), "participation_loss": float(p_loss.detach()), "candidate_loss": float(c_loss.detach()),
            "own_commit_loss": float(own_loss.detach()), "counterfactual_commit_loss": float(cf_loss.detach()),
            "decision_nodes": int(mask.sum())}


@torch.inference_mode()
def validate_local(model, examples, config, device):
    """Local accuracies on independent validation rounds of the training degrees."""
    model.eval()
    per_degree = {}
    for degree in config.degrees:
        subset = [e for e in examples if e.degree == degree]
        totals = dict(nodes=0, p_correct=0, c_correct=0, commit_correct_own=0, action_correct_teacher=0)
        loss_sum = 0.0
        for start in range(0, len(subset), config.batch_size):
            batch = make_batch(subset[start:start + config.batch_size], device)
            out = model(*batch.inputs())
            mask = batch.mask
            own_target = torch.from_numpy(exact_commit_labels(_np(batch.colors), _np(batch.edge_index), _np(out.participate),
                                                              _np(out.candidate)).astype(np.float32)).to(device)
            n = int(mask.sum())
            loss = (F.binary_cross_entropy_with_logits(out.participation_logit[mask], batch.y_participate[mask])
                    + F.cross_entropy(out.candidate_logits[mask], batch.y_offer[mask])
                    + F.binary_cross_entropy_with_logits(out.commit_logit[mask], own_target[mask]))
            action = torch.where(out.participate & (out.candidate >= 0) & (out.commit_logit > 0), out.candidate, -1)
            truth = torch.where(batch.y_commit > 0.5, batch.y_offer, -1)
            totals["nodes"] += n
            totals["p_correct"] += int(((out.participate == (batch.y_participate > 0.5)) & mask).sum())
            totals["c_correct"] += int(((out.candidate == batch.y_offer) & mask).sum())
            totals["commit_correct_own"] += int((((out.commit_logit > 0) == (own_target > 0.5)) & mask).sum())
            totals["action_correct_teacher"] += int(((action == truth) & mask).sum())
            loss_sum += float(loss) * n
        per_degree[str(degree)] = {**totals, "loss": loss_sum / totals["nodes"],
                                   **{k + "_rate": v / totals["nodes"] for k, v in totals.items() if k != "nodes"}}
    nodes = sum(v["nodes"] for v in per_degree.values())
    pooled = {"nodes": nodes, "loss": sum(v["loss"] * v["nodes"] for v in per_degree.values()) / nodes}
    for key in ("p_correct", "c_correct", "commit_correct_own", "action_correct_teacher"):
        pooled[key + "_rate"] = sum(v[key] for v in per_degree.values()) / nodes
    return {**pooled, "per_degree": per_degree}


def validate_rollouts(model, config):
    """Replica-specific validation rollouts (the M6B streams) for training degrees only."""
    rows = []
    for degree in config.degrees:
        counts = dict.fromkeys(config.val_sizes, config.val_rollouts)
        for metadata in cohort_cases(degree, config.val_sizes, counts, split="validation_rollout", namespace=config.seed):
            edges, info = build_case(metadata)
            result, _ = run_rollout(model, metadata["n"], edges, degree + 1, metadata["tape_seed"], config.max_rounds)
            result.pop("by_k")
            rows.append({**metadata, **info, **result})
    successes = sum(r["success"] for r in rows)
    states = sum(r["states"] for r in rows)
    return {"successes": successes, "graphs": len(rows), "success_rate": successes / len(rows),
            "unsafe_commits": sum(r["unsafe_commits"] for r in rows),
            "unsafe_per_state": sum(r["unsafe_commits"] for r in rows) / max(1, states),
            "mean_rounds": float(np.mean([r["rounds"] for r in rows])), "per_graph": rows}


def train(config, root, device="cuda", stop_requested=lambda: False, stop_after_epoch=None, check_resources=True):
    """Caller holds the GPU lock. An interrupted epoch repeats from last.pt."""
    root = Path(root)
    run_dir = root / "runs" / config.id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and digest(json.loads(config_path.read_text())) != digest(config.to_dict()):
        raise RuntimeError("Existing run configuration differs; overwrite refused")
    atomic_json(config_path, config.to_dict())
    model, optimizer, order, cf_rng = initialize(config, device)
    snapshot_resources = (lambda: resources(root, device=device)) if check_resources else (lambda: {})
    saved = None
    if (run_dir / "last.pt").exists():
        saved = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if saved["config"] != config.to_dict():
            raise RuntimeError("Checkpoint configuration differs")
        current = provenance()
        if current["source_hashes"] != saved["provenance"]["source_hashes"]:
            raise RuntimeError("Source changed since checkpoint; exact resume refused")
        if current["versions"] != saved["provenance"]["versions"] or saved["device"] != str(device):
            raise RuntimeError("Runtime versions/device changed; exact resume refused")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if saved["complete"]:
            atomic_json(run_dir / "history.json", saved["history"])
            return "complete"
    snapshot_resources()
    print(f"{config.id} generating data on {device}", flush=True)
    tick = time.perf_counter()
    train_examples, train_records = collect_examples(config, "train")
    val_examples, val_records = collect_examples(config, "validation")
    data_generation_seconds = time.perf_counter() - tick
    if {r["fingerprint"] for r in train_records} & {r["fingerprint"] for r in val_records}:
        raise RuntimeError("Training and validation graphs overlap")
    manifest = {"training_rounds": len(train_examples), "validation_rounds": len(val_examples),
                "training_trajectories": sum(config.graphs_per_degree) * config.tapes,
                "train": train_records, "validation": val_records}
    data_hash = digest(manifest)
    manifest_path = run_dir / "data_manifest.json"
    if manifest_path.exists() and digest(json.loads(manifest_path.read_text())) != data_hash:
        raise RuntimeError("Regenerated data differs from persistent manifest")
    atomic_json(manifest_path, manifest)
    history, best_score, updates, rounds_seen, nodes_seen, elapsed = [], None, 0, 0, 0, 0.0
    metadata = provenance()
    initial_hash = digest({k: v.detach().cpu().tolist() for k, v in model.state_dict().items()})
    if saved:
        if saved["data_hash"] != data_hash:
            raise RuntimeError("Dataset differs from checkpoint")
        history, best_score = saved["history"], saved["best_score"]
        updates, rounds_seen, nodes_seen = saved["updates"], saved["rounds_seen"], saved["nodes_seen"]
        elapsed, metadata, initial_hash = saved["elapsed_seconds"], saved["provenance"], saved["initial_state_hash"]
        data_generation_seconds = saved["data_generation_seconds"]
        restore_rng(saved["rng"], order, cf_rng)
    parameters = count_parameters(model)
    print(f"parameters={parameters} rounds={len(train_examples)} resume_epoch={len(history)}", flush=True)
    for epoch in range(len(history) + 1, config.epochs + 1):
        if stop_requested():
            return "paused"
        snapshot = snapshot_resources()
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tick, losses, weighted = time.perf_counter(), {}, 0
        permutation = order.permutation(len(train_examples))
        for start in range(0, len(permutation), config.batch_size):
            examples = [train_examples[int(i)] for i in permutation[start:start + config.batch_size]]
            metrics = training_step(model, optimizer, make_batch(examples, device), config, epoch, cf_rng)
            n = metrics.pop("decision_nodes")
            for key, value in metrics.items():
                losses[key] = losses.get(key, 0.0) + value * n
            weighted += n
            updates += 1
            rounds_seen += len(examples)
            nodes_seen += n
        train_seconds = time.perf_counter() - tick
        local = validate_local(model, val_examples, config, device)
        rollouts, is_best = None, False
        if epoch == 1 or epoch % config.val_every == 0 or epoch == config.epochs:
            rollouts = validate_rollouts(model, config)
            score = (rollouts["success_rate"], -rollouts["unsafe_per_state"], -rollouts["mean_rounds"],
                     local["action_correct_teacher_rate"], -local["loss"])
            if best_score is None or score > tuple(best_score):
                best_score, is_best = list(score), True
        elapsed += time.perf_counter() - tick
        peak = torch.cuda.max_memory_allocated() if device != "cpu" and torch.cuda.is_available() else 0
        row = {"epoch": epoch, "updates": updates, "rounds_seen": rounds_seen, "nodes_seen": nodes_seen,
               "train": {k: v / weighted for k, v in losses.items()}, "validation_local": local,
               "validation_rollouts": rollouts, "train_seconds": train_seconds, "elapsed_seconds": elapsed,
               "gpu_peak_bytes": int(peak), "resources": snapshot, "data_generation_seconds": data_generation_seconds}
        history.append(row)
        checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch, "updates": updates, "rounds_seen": rounds_seen, "nodes_seen": nodes_seen,
                      "config": config.to_dict(), "rng": rng_state(order, cf_rng), "data_hash": data_hash,
                      "initial_state_hash": initial_hash, "history": history, "best_score": best_score,
                      "elapsed_seconds": elapsed, "parameters": parameters, "provenance": metadata,
                      "data_generation_seconds": data_generation_seconds, "device": str(device),
                      "complete": epoch == config.epochs}
        if is_best:
            save_checkpoint(run_dir / "best.pt", checkpoint)
        save_checkpoint(run_dir / "last.pt", checkpoint)
        if epoch in config.checkpoint_epochs:
            path = run_dir / f"epoch-{epoch:03d}.pt"
            if path.exists():
                raise RuntimeError("Refusing to overwrite a historical epoch checkpoint")
            save_checkpoint(path, checkpoint)
        atomic_json(run_dir / "history.json", history)
        message = (f"epoch={epoch:03d} loss={row['train']['loss']:.5f} p={local['p_correct_rate']:.5f} "
                   f"c={local['c_correct_rate']:.5f} commit={local['commit_correct_own_rate']:.5f} "
                   f"action={local['action_correct_teacher_rate']:.5f} seconds={elapsed:.1f}")
        if rollouts:
            message += f" val_success={rollouts['successes']}/{rollouts['graphs']} unsafe={rollouts['unsafe_commits']}"
        print(message, flush=True)
        if (stop_after_epoch is not None and epoch >= stop_after_epoch and not checkpoint["complete"]) or stop_requested():
            if not checkpoint["complete"]:
                return "paused"
    return "complete"


def load_model(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    config = saved["config"]
    model = UniversalColorGNN(config["hidden_dim"], config["message_dim"], config["variant"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, saved
