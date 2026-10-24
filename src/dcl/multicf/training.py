"""Fixed-bank controls with exact Adam/RNG/visit-count resumption."""
from collections import Counter
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from dcl.multiteacher.programs import DETECT, LINIAL, MAX_FIELD, named_seed
from dcl.multiteacher.experiment import weight_hash
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash, save_checkpoint
from .data import load
from .oracles import confirmation
from .protocol import cases, source_hashes, summarize_runs

METHODS = ("passive", "matched", "counterfactual", "owncf")


def objective(model, packed, positions, method, update, warmup, variant, query_cache=None):
    arrays = tuple(packed[f"a{i}"][positions] for i in range(4))
    native = packed["native"][positions]
    phase = int(native[0, 3])
    out = model(*tensor_arrays(arrays), phase=phase)
    z = out.commit.new_zeros(())
    if phase == LINIAL:
        ce = F.cross_entropy(out.scores, torch.from_numpy(native[:, 1] // MAX_FIELD))
        return ce, dict(point=float(ce.detach())), 0
    if phase == DETECT:
        bce = F.binary_cross_entropy_with_logits(out.commit, torch.from_numpy(native[:, 2].astype(np.float32)))
        return bce, dict(mark=float(bce.detach())), 0
    p, c = packed["p"][positions], packed["c"][positions]
    pl = F.binary_cross_entropy_with_logits(out.state.p_logits[:, 0], torch.from_numpy(p[:, 0].astype(np.float32)))
    cl = F.cross_entropy(out.scores, torch.from_numpy(c[:, 0]))
    teacher_y = torch.from_numpy((native[:, 0] > 0).astype(np.float32))
    queries = 0
    if method == "passive":
        native_loss = cf_loss = F.binary_cross_entropy_with_logits(out.commit, teacher_y)
    elif method == "matched" or update < warmup:
        logits = model.commit_for(out.state, torch.from_numpy(p), torch.from_numpy(c))
        native_loss = cf_loss = F.binary_cross_entropy_with_logits(logits, teacher_y)
    else:
        if method == "owncf":
            op, oc = out.p.detach().numpy(), out.candidates.detach().numpy()
            # Every new student-specific query is charged. Cache key includes
            # stored state identity and all four actual P/C messages.
            keys = [(int(i), tuple(map(bool, pp)), tuple(map(int, cc))) for i, pp, cc in zip(positions, op, oc)]
            truth = np.empty(len(keys), np.float32)
            for j, key in enumerate(keys):
                if key not in query_cache:
                    value = float(confirmation(tuple(a[j:j+1] for a in arrays), model.teacher, op[j:j+1], oc[j:j+1])[0])
                    query_cache[key] = value; queries += 1
                truth[j] = query_cache[key]
            native_loss = F.binary_cross_entropy_with_logits(out.commit, torch.from_numpy(truth))
        else:
            logits = model.commit_for(out.state, torch.from_numpy(p), torch.from_numpy(c))
            native_loss = F.binary_cross_entropy_with_logits(logits, teacher_y)
        cp = packed["cf_p"][positions, variant]
        cc = packed["cf_c"][positions, variant]
        cy = packed["cf_y"][positions, variant]
        logits = model.commit_for(out.state, torch.from_numpy(cp), torch.from_numpy(cc))
        cf_loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(cy))
    loss = pl + cl + .5 * native_loss + .5 * cf_loss
    return loss, dict(participation=float(pl.detach()), candidate=float(cl.detach()),
        native_or_own_confirmation=float(native_loss.detach()), counterfactual_confirmation=float(cf_loss.detach())), queries


def train(config, teacher, architecture, seed, dataset, directory, method="counterfactual", updates=None,
          checkpoints=None, stop_after=None, development=True):
    if method not in METHODS:
        raise ValueError(method)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    updates = updates or config["updates"]
    checkpoints = sorted(set(checkpoints or config["checkpoints"]) | {updates})
    packed, manifest = load(dataset)
    identity = dict(namespace=config["namespace"], teacher=teacher, architecture=architecture, seed=seed,
        method=method, updates=updates, checkpoints=checkpoints, batch_size=config["batch_size"],
        lr=config["lr"], warmup=config["warmup_updates"], data_sha256=manifest["sha256"])
    atomic_json(directory / "config.json", dict(config=config, identity=identity)) if not (directory / "config.json").exists() else None
    if json.loads((directory / "config.json").read_text())["identity"] != identity:
        raise RuntimeError("Training identity changed")
    model = initialize(teacher, architecture, seed)
    initial_hash = weight_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    order = np.random.default_rng(named_seed(config["namespace"], "batch-order", seed))
    phases = sorted(np.unique(packed["native"][:, 3]))
    populations = {int(p): np.flatnonzero(packed["native"][:, 3] == p) for p in phases}
    visits = np.zeros(len(packed["native"]), np.int64)
    cf_visits = np.zeros((len(visits), 4), np.int64)
    history, presentations, query_cache = [], Counter(), {}
    elapsed = cpu_elapsed = 0.
    start = 0
    sources = source_hashes()
    last = directory / "last.pt"
    if last.exists():
        saved = torch.load(last, weights_only=False)
        if saved["identity"] != identity or saved["sources"] != sources:
            raise RuntimeError("Resume identity/source mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        order.bit_generator.state = saved["batch_rng"]
        torch.set_rng_state(saved["torch_rng"]); random.setstate(saved["python_rng"])
        visits, cf_visits = saved["visits"], saved["cf_visits"]
        history, presentations, query_cache = saved["history"], Counter(saved["presentations"]), saved["query_cache"]
        start, elapsed, cpu_elapsed = saved["update"], saved["elapsed"], saved["cpu_elapsed"]
        # last.pt is authoritative after a crash between checkpoint and report.
        if start in checkpoints:
            target = directory / f"u{start:06d}.pt"
            if not target.exists():
                save_checkpoint(target, saved)
            record_path = directory / f"u{start:06d}-record.json"
            if not record_path.exists():
                record = dict(saved["record"], checkpoint_sha256=file_hash(target))
                if development:
                    suite = cases(config, "development", seed)
                    rows, seconds = evaluate(model, teacher, suite)
                    atomic_json(directory / f"u{start:06d}-development.json", rows)
                    record["development"] = summarize_runs(rows, [m["case_id"] for _, m in suite])
                    record["development_seconds"] = seconds
                atomic_json(record_path, record)
    tick, cpu = time.perf_counter(), time.process_time()
    for u in range(start, updates):
        phase = int(phases[u % len(phases)])
        positions = order.choice(populations[phase], config["batch_size"], replace=True)
        # Same variant sequence and native minibatches in every paired arm.
        variant = (u // len(phases)) % 4
        np.add.at(visits, positions, 1)
        for c in (("linial_point",) if phase == LINIAL else (("conflict_mark",) if phase == DETECT else
                  ("participation", "candidate", "confirmation"))):
            presentations[c] += len(positions)
        if method in ("counterfactual", "owncf") and u >= config["warmup_updates"] and phase not in (LINIAL, DETECT):
            np.add.at(cf_visits, (positions, np.full(len(positions), variant)), 1)
            presentations["counterfactual_confirmation"] += len(positions)
        optimizer.zero_grad(set_to_none=True)
        loss, parts, q = objective(model, packed, positions, method, u, config["warmup_updates"], variant, query_cache)
        presentations["new_own_confirmation_queries"] += q
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite loss")
        loss.backward(); optimizer.step()
        if (u+1) % config["log_every"] == 0 or u+1 in checkpoints:
            history.append(dict(update=u+1, loss=float(loss.detach()), **parts))
        must_save = (u+1) % config["save_every"] == 0 or u+1 in checkpoints or u+1 == stop_after
        if must_save:
            wall, proc = elapsed+time.perf_counter()-tick, cpu_elapsed+time.process_time()-cpu
            acquired = manifest["native_labels"] + (manifest["counterfactual_labels"] if method in ("counterfactual", "owncf") else 0) + len(query_cache)
            record = dict(identity=identity, update=u+1, initial_hash=initial_hash, weights_hash=weight_hash(model),
                parameters=sum(p.numel() for p in model.parameters()), native_labels=manifest["native_labels"],
                counterfactual_labels=manifest["counterfactual_labels"] if method in ("counterfactual", "owncf") else 0,
                own_confirmation_labels=len(query_cache), total_acquired_component_labels=acquired,
                native_examples_used=int((visits>0).sum()), counterfactual_labels_used=int((cf_visits>0).sum()),
                component_presentations=dict(presentations), history=history, training_seconds=wall, cpu_seconds=proc,
                gpu_seconds=0., device="cpu", peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                sources=sources, data_manifest_sha256=file_hash(Path(dataset)/"manifest.json"))
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), identity=identity,
                initial_hash=initial_hash, update=u+1, torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                batch_rng=order.bit_generator.state, visits=visits, cf_visits=cf_visits, history=history,
                presentations=dict(presentations), query_cache=query_cache, sources=sources, elapsed=wall,
                cpu_elapsed=proc, record=record)
            save_checkpoint(last, payload)
            if u+1 in checkpoints:
                target = directory / f"u{u+1:06d}.pt"
                if target.exists():
                    raise RuntimeError("Refusing to overwrite frozen intermediate checkpoint")
                save_checkpoint(target, payload)
                record["checkpoint_sha256"] = file_hash(target)
                if development:
                    suite = cases(config, "development", seed)
                    rows, seconds = evaluate(model, teacher, suite)
                    atomic_json(directory / f"u{u+1:06d}-development.json", rows)
                    record["development"] = summarize_runs(rows, [m["case_id"] for _,m in suite])
                    record["development_seconds"] = seconds
                atomic_json(directory / f"u{u+1:06d}-record.json", record)
                print(json.dumps(dict(teacher=teacher, architecture=architecture, seed=seed, method=method,
                    update=u+1, seconds=round(wall, 2), development=record.get("development", {}).get("successes"))), flush=True)
                model.train()
            # Evaluation time is separate from optimization time.
            elapsed, cpu_elapsed = wall, proc
            tick, cpu = time.perf_counter(), time.process_time()
        if u+1 == stop_after:
            break
    completed = (u+1) if start < updates else start
    return dict(directory=str(directory), identity=identity, complete=(completed == updates))
