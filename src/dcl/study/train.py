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

from dcl.counterfactual import commit_logits_for_proposals, exact_commit_labels, make_counterfactual_proposals
from dcl.m4_models import M4B
from .config import seed_for, digest
from .data import collect_examples, make_batch
from .evaluation import validate_rollouts
from .storage import atomic_json, provenance, resources, save_checkpoint, Registry


def initialize(config, device):
    seed = seed_for(config.seed, config.degree, "initialization")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = M4B(config.hidden_dim, config.message_dim, config.num_colors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    order = np.random.default_rng(seed_for(config.seed, config.degree, "minibatches"))
    counterfactual = np.random.default_rng(seed_for(config.seed, config.degree, "counterfactual"))
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


def training_step(model, optimizer, batch, config, epoch, cf_rng):
    x, edges, yp, yc, teacher_commit, mask = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    traditional_warmup = config.method == "TRADITIONAL" and epoch <= config.warmup
    p, c, own_logits = model(x, edges, hard_messages=not traditional_warmup)
    p_loss = F.binary_cross_entropy_with_logits(p[mask], yp[mask])
    c_loss = F.cross_entropy(c[mask], yc[mask])
    if traditional_warmup:
        own_loss = cf_loss = p_loss.new_zeros(())
    elif config.method == "TRADITIONAL":
        # Deliberate historical reference: labels for the teacher's proposals
        # applied to the model's proposals. Kept separate and clearly named.
        own_loss = cf_loss = F.binary_cross_entropy_with_logits(own_logits[mask], teacher_commit[mask])
    elif epoch <= config.warmup or config.method == "ORACLE_MATCHED":
        expert = commit_logits_for_proposals(model, x, edges, yp > 0.5, yc)
        own_loss = cf_loss = F.binary_cross_entropy_with_logits(expert[mask], teacher_commit[mask])
    else:
        x_np, e_np = x.detach().cpu().numpy(), edges.detach().cpu().numpy()
        p_np, c_np = (p.detach() > 0).cpu().numpy(), c.detach().argmax(1).cpu().numpy()
        own_target = torch.from_numpy(exact_commit_labels(x_np, e_np, p_np, c_np)).to(x.device)
        own_loss = F.binary_cross_entropy_with_logits(own_logits[mask], own_target[mask])
        if config.method == "COUNTERFACTUAL":
            cf_p, cf_c = make_counterfactual_proposals(x_np, e_np, yp.cpu().numpy(), yc.cpu().numpy(), cf_rng)
            cf_target = torch.from_numpy(exact_commit_labels(x_np, e_np, cf_p, cf_c)).to(x.device)
        else:
            # Identical update budget, two COMMIT evaluations, no artificial
            # proposals. The 0.5 weights avoid doubling the loss scale.
            cf_p, cf_c, cf_target = p_np, c_np, own_target
        cf_logits = commit_logits_for_proposals(model, x, edges, torch.as_tensor(cf_p, device=x.device), torch.as_tensor(cf_c, device=x.device))
        cf_loss = F.binary_cross_entropy_with_logits(cf_logits[mask], cf_target[mask])
    loss = p_loss + c_loss + 0.5 * own_loss + 0.5 * cf_loss
    loss.backward()
    if not traditional_warmup:
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
    model.eval()
    totals = dict(nodes=0, p_correct=0, c_correct=0, commit_correct_own=0, action_correct_teacher=0)
    loss_sum = 0.0
    for start in range(0, len(examples), config.batch_size):
        x, edges, yp, yc, teacher_commit, mask = make_batch(examples[start:start + config.batch_size], device)
        p, c, commit = model(x, edges, hard_messages=True)
        p_pred, c_pred = p > 0, c.argmax(1)
        own_target = torch.from_numpy(exact_commit_labels(x.cpu().numpy(), edges.cpu().numpy(), p_pred.cpu().numpy(), c_pred.cpu().numpy())).to(device)
        n = int(mask.sum())
        loss = (F.binary_cross_entropy_with_logits(p[mask], yp[mask]) + F.cross_entropy(c[mask], yc[mask])
                + F.binary_cross_entropy_with_logits(commit[mask], own_target[mask]))
        action = torch.where(p_pred & (c_pred > 0) & (commit > 0), c_pred, 0)
        truth_action = torch.where(teacher_commit > 0.5, yc, 0)
        totals["nodes"] += n
        totals["p_correct"] += int(((p_pred == (yp > 0.5)) & mask).sum())
        totals["c_correct"] += int(((c_pred == yc) & mask).sum())
        totals["commit_correct_own"] += int((((commit > 0) == (own_target > 0.5)) & mask).sum())
        totals["action_correct_teacher"] += int(((action == truth_action) & mask).sum())
        loss_sum += float(loss) * n
    return {**totals, "loss": loss_sum / totals["nodes"],
            **{key + "_rate": value / totals["nodes"] for key, value in totals.items() if key != "nodes"}}


def train(config, root, device="cuda", stop_requested=lambda: False, stop_after_epoch=None):
    """Caller holds the global lock. Interrupted epochs repeat from last.pt."""
    root = Path(root)
    run_dir = root / "runs" / config.id
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = Registry(root)
    config_path = run_dir / "config.json"
    if config_path.exists() and digest(json.loads(config_path.read_text())) != digest(config.to_dict()):
        raise RuntimeError("Existing run configuration differs; overwrite refused")
    atomic_json(config_path, config.to_dict())
    model, optimizer, order, cf_rng = initialize(config, device)
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
            if saved["epoch"] in config.checkpoint_epochs and not (run_dir / f"epoch-{saved['epoch']:03d}.pt").exists():
                save_checkpoint(run_dir / f"epoch-{saved['epoch']:03d}.pt", saved)
            registry.set(config, "complete", saved["epoch"], saved["updates"])
            return "complete"
        if saved["epoch"] in config.checkpoint_epochs and not (run_dir / f"epoch-{saved['epoch']:03d}.pt").exists():
            save_checkpoint(run_dir / f"epoch-{saved['epoch']:03d}.pt", saved)
    resources(root, device=device)
    registry.set(config, "generating", saved["epoch"] if saved else 0, saved["updates"] if saved else 0)
    print(f"{config.id} generating data on {device}", flush=True)
    generation_tick = time.perf_counter()
    train_examples, train_records = collect_examples(config, "train")
    val_examples, val_records = collect_examples(config, "validation")
    data_generation_seconds = time.perf_counter() - generation_tick
    if {r["fingerprint"] for r in train_records} & {r["fingerprint"] for r in val_records}:
        raise RuntimeError("Training and validation graphs overlap")
    data_manifest = {"dataset_id": config.dataset_id, "training_rounds": len(train_examples), "validation_rounds": len(val_examples),
                     "training_trajectories": config.graphs * config.tapes, "train": train_records, "validation": val_records}
    data_hash = digest(data_manifest)
    manifest_path = run_dir / "data_manifest.json"
    if manifest_path.exists() and digest(json.loads(manifest_path.read_text())) != data_hash:
        raise RuntimeError("Regenerated data differs from persistent manifest")
    atomic_json(manifest_path, data_manifest)
    history, best_score, updates, rounds_seen, nodes_seen, elapsed_seconds = [], None, 0, 0, 0, 0.0
    metadata = provenance()
    initial_state_hash = digest({name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()})
    if saved:
        if saved["data_hash"] != data_hash:
            raise RuntimeError("Dataset differs from checkpoint")
        history, best_score = saved["history"], saved["best_score"]
        updates, rounds_seen, nodes_seen = saved["updates"], saved["rounds_seen"], saved["nodes_seen"]
        elapsed_seconds, metadata = saved["elapsed_seconds"], saved["provenance"]
        initial_state_hash = saved["initial_state_hash"]
        data_generation_seconds = saved["data_generation_seconds"]
        restore_rng(saved["rng"], order, cf_rng)
    print(f"parameters={sum(p.numel() for p in model.parameters())} rounds={len(train_examples)} resume_epoch={len(history)}", flush=True)
    for epoch in range(len(history) + 1, config.epochs + 1):
        if stop_requested():
            registry.set(config, "paused", epoch - 1, updates)
            return "paused"
        snapshot = resources(root, device=device)
        tick, losses, weighted_nodes = time.perf_counter(), {}, 0
        steps = config.updates_per_epoch or math.ceil(len(train_examples) / config.batch_size)
        permutation, position = order.permutation(len(train_examples)), 0
        for _ in range(steps):
            if position >= len(permutation):
                permutation, position = order.permutation(len(train_examples)), 0
            indices = permutation[position:position + config.batch_size]
            position += len(indices)
            examples = [train_examples[int(i)] for i in indices]
            metrics = training_step(model, optimizer, make_batch(examples, device), config, epoch, cf_rng)
            n = metrics.pop("decision_nodes")
            for key, value in metrics.items():
                losses[key] = losses.get(key, 0.0) + value * n
            weighted_nodes += n
            updates += 1
            rounds_seen += len(examples)
            nodes_seen += n
        local = validate_local(model, val_examples, config, device)
        rollouts = None
        is_best = False
        if epoch == 1 or epoch % config.val_every == 0 or epoch == config.epochs:
            rollouts = validate_rollouts(model, config)
            score = (rollouts["success_rate"], -rollouts["unsafe_per_state"], -rollouts["mean_rounds"], local["action_correct_teacher_rate"], -local["loss"])
            if best_score is None or score > tuple(best_score):
                best_score, is_best = score, True
        elapsed_seconds += time.perf_counter() - tick
        row = {"epoch": epoch, "updates": updates, "rounds_seen": rounds_seen, "nodes_seen": nodes_seen,
               "train": {k: v / weighted_nodes for k, v in losses.items()}, "validation_local": local,
               "validation_rollouts": rollouts, "elapsed_seconds": elapsed_seconds, "resources": snapshot}
        row["data_generation_seconds"] = data_generation_seconds
        history.append(row)
        checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch, "updates": updates, "rounds_seen": rounds_seen, "nodes_seen": nodes_seen,
                      "config": config.to_dict(), "rng": rng_state(order, cf_rng), "data_hash": data_hash,
                      "dataset_id": config.dataset_id, "initial_state_hash": initial_state_hash,
                      "history": history, "best_score": best_score, "elapsed_seconds": elapsed_seconds,
                      "parameters": sum(p.numel() for p in model.parameters()), "provenance": metadata,
                      "data_generation_seconds": data_generation_seconds, "device": str(device),
                      "complete": epoch == config.epochs}
        # last.pt is authoritative. Other epoch outputs can be reconstructed
        # from its history after a crash; no append-only log is trusted as state.
        if is_best:
            save_checkpoint(run_dir / "best.pt", checkpoint)
        save_checkpoint(run_dir / "last.pt", checkpoint)
        if epoch in config.checkpoint_epochs:
            checkpoint_path = run_dir / f"epoch-{epoch:03d}.pt"
            if checkpoint_path.exists():
                raise RuntimeError("Refusing to overwrite a historical epoch checkpoint")
            save_checkpoint(checkpoint_path, checkpoint)
        atomic_json(run_dir / "history.json", history)
        registry.set(config, "complete" if checkpoint["complete"] else "running", epoch, updates)
        print(f"epoch={epoch:03d} loss={row['train']['loss']:.5f} val_action={local['action_correct_teacher_rate']:.6f} updates={updates} seconds={elapsed_seconds:.1f}", flush=True)
        if (stop_after_epoch is not None and epoch >= stop_after_epoch) or stop_requested():
            if not checkpoint["complete"]:
                registry.set(config, "paused", epoch, updates)
                return "paused"
    return "complete"
