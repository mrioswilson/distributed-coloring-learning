"""Budget-matched acquisition/training and a freeze-before-test driver.

Run from repository root. Every output is new; no historical study is mutated.
Final evaluation is deliberately a separate, guarded stage.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dcl.multicf.data import training_graph
from dcl.multicf.oracles import confirmation, stage_targets
from dcl.multicf.protocol import cases
from dcl.multicf.states import trajectory
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import LINIAL, DETECT, KW, ASSIGN, REPAIR, MAX_FIELD, named_seed
from dcl.multiteacher.experiment import weight_hash
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.data import array_hash, components
from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.protocol import summarize_runs
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash, now, save_checkpoint
from .keys import commit_key, key_text, held_out

ROOT = Path("results/paper-revision-20261030")
CONFIG = Path("configs/paper-revision-20261030.json")


def sources():
    paths = list(Path("src/dcl/paperrevision").glob("*.py"))
    paths += list(Path("src/dcl/multicf").glob("*.py"))
    paths += list(Path("src/dcl/multiteacher").glob("*.py"))
    paths += list(Path("src/dcl/multiteacher_e1").glob("*.py"))
    paths += list(Path("src/dcl/universal").glob("*.py"))
    paths += [Path("src/dcl/samplecomplexity") / p for p in
              ("model.py", "evaluation.py", "protocol.py", "training.py")]
    paths += [Path("src/dcl/integration/models.py"), Path("src/dcl/paperstudy/models.py"), CONFIG]
    return {str(p): file_hash(p) for p in sorted(paths)}


def intervention(p, c, allowed, indices, rng):
    pp, cc = p.copy(), c.copy()
    variant = indices % 4
    cc[:, 0] = (cc[:, 0] + variant) % 4
    pp[:, 0] = allowed[:, 0]
    pp[(variant == 3) & (rng.random(len(pp)) < .25), 0] = False
    return pp & allowed, cc


def stream(config, teacher, seed, kind):
    local = dict(config, namespace=config["namespace"] + ":" + kind)
    if kind == "diagnostic":
        local["training_sizes"] = config["diagnostic_sizes"]
    t = 0
    while True:
        graph, family, gs = training_graph(local, seed, t, True)
        tape = named_seed(local["namespace"], seed, "tape", t)
        for program, state, exact, witness in trajectory(graph, teacher, tape):
            roots = np.flatnonzero(state.permission)
            if not len(roots):
                continue
            arrays = encode_roots(state, program.neighbors, roots)
            native = np.column_stack((exact.color[roots], exact.auxiliary[roots],
                                      exact.mark[roots], np.full(len(roots), state.phase)))
            if state.phase not in (LINIAL, DETECT):
                targets = stage_targets(arrays, native, teacher)
                p, c = targets.participate, targets.candidate
                y = confirmation(arrays, teacher, p, c)
                if not np.array_equal(y > .5, native[:, 0] > 0):
                    raise RuntimeError("Native/oracle decomposition disagrees")
            else:
                p = np.zeros((len(roots), 4), bool)
                c = np.zeros((len(roots), 4), np.int64)
            yield arrays, native, p, c, roots, t, program.steps, dict(
                **witness, family=family, graph_seed=gs)
        t += 1


def pack(buffers):
    return {k: np.concatenate(v) for k, v in buffers.items()}


def retain(buffers, arrays, native, p, c, use, roots, t, step, **extra):
    values = {f"a{j}": a for j, a in enumerate(arrays)}
    values.update(native=native, p=p, c=c, node=roots,
                  trajectory=np.full(len(native), t), step=np.full(len(native), step), **extra)
    for k, a in values.items():
        buffers[k].append(a[use])


def acquire(config, teacher, seed, directory, kind="train"):
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, sha in manifest["files"].items():
            if file_hash(directory / name) != sha:
                raise RuntimeError("Dataset hash mismatch")
        return manifest
    rng = np.random.default_rng(named_seed(config["namespace"], teacher, seed, kind, "intervention"))
    tick, cpu = time.perf_counter(), time.process_time()
    stats = Counter(); witnesses = []
    base, aux = defaultdict(list), defaultdict(list)
    keys = set(); query_hashes = {"native": set(), "cf": set()}
    phase_counts = Counter()
    if kind == "train":
        cost = 0
        for arrays, native, p, c, roots, t, step, witness in stream(config, teacher, seed, "base"):
            phase = int(native[0, 3]); unit = len(components(phase))
            stats["base_teacher_simulator_steps_observed"] += 1
            stats["base_root_contexts_considered"] += len(native)
            allowed = np.ones(len(native), bool)
            if phase not in (LINIAL, DETECT):
                rowkeys = [commit_key(arrays, i, p, c, teacher) for i in range(len(native))]
                allowed = np.array([not held_out(k) for k in rowkeys])
                stats["base_type_filter_rejections"] += int((~allowed).sum())
            use = np.flatnonzero(allowed)[:(config["native_budget"] - cost) // unit]
            if not len(use):
                continue
            retain(base, arrays, native, p, c, use, roots, t, step)
            cost += unit * len(use)
            for comp in components(phase):
                stats["base_target_" + comp] += len(use)
            if phase not in (LINIAL, DETECT):
                keys.update(key_text(rowkeys[i]) for i in use)
            if len(witnesses) < 200:
                witnesses.append(dict(**witness, bank="base", root=int(roots[use[0]])))
            if cost == config["native_budget"]:
                break
            if t > 10000:
                raise RuntimeError("Native budget not filled")
        np.savez_compressed(directory / "base.npz", **pack(base))
        phases = [KW] if teacher == "BE_BASE_D3" else [ASSIGN, REPAIR]
        targets = {p: config["auxiliary_budget"] // len(phases) for p in phases}
    else:
        phases = [KW] if teacher == "BE_BASE_D3" else [ASSIGN, REPAIR]
        targets = {p: config["diagnostic_queries"] // len(phases) for p in phases}
    cursor = 0
    for arrays, native, p, c, roots, t, step, witness in stream(config, teacher, seed,
                                                             "auxiliary" if kind == "train" else "diagnostic"):
        phase = int(native[0, 3]); stats["aux_teacher_simulator_steps_observed"] += 1
        if phase not in targets or phase_counts[phase] >= targets[phase]:
            continue
        cp, cc = intervention(p, c, arrays[0][..., 7] > .5,
                              np.arange(cursor, cursor + len(native)), rng)
        cursor += len(native)
        nk = [commit_key(arrays, i, p, c, teacher) for i in range(len(native))]
        ck = [commit_key(arrays, i, cp, cc, teacher) for i in range(len(native))]
        allowed = np.array([(not held_out(a) and not held_out(b)) if kind == "train" else held_out(b)
                            for a, b in zip(nk, ck)])
        stats["aux_root_contexts_considered"] += len(native)
        stats["aux_type_filter_rejections"] += int((~allowed).sum())
        use = np.flatnonzero(allowed)[:targets[phase] - phase_counts[phase]]
        if not len(use):
            continue
        cy = confirmation(tuple(a[use] for a in arrays), teacher, cp[use], cc[use])
        ny = confirmation(tuple(a[use] for a in arrays), teacher, p[use], c[use])
        # Labels acquired only after the predetermined type filter, never used to select states.
        fullcy = np.zeros(len(native), np.float32); fullcy[use] = cy
        fullny = np.zeros(len(native), np.float32); fullny[use] = ny
        retain(aux, arrays, native, p, c, use, roots, t, step,
               cf_p=cp, cf_c=cc, cf_y=fullcy, native_y=fullny)
        phase_counts[phase] += len(use)
        stats["conditioning_root_values"] += 2 * len(use)
        stats["conditioning_real_center_values"] += int(2 * (1 + arrays[3][use, 0].sum(1)).sum())
        for i in use:
            if kind == "train":
                keys.update((key_text(nk[i]), key_text(ck[i])))
            else:
                keys.add(key_text(ck[i]))
            for arm, pp, qq in (("native", p, c), ("cf", cp, cc)):
                query_hashes[arm].add(array_hash(*(a[i:i+1] for a in arrays), pp[i:i+1], qq[i:i+1]))
        if len(witnesses) < 400:
            witnesses.append(dict(**witness, bank=kind, root=int(roots[use[0]])))
        if all(phase_counts[p] == targets[p] for p in targets):
            break
        if t > 20000:
            raise RuntimeError("Auxiliary budget not filled")
    np.savez_compressed(directory / "aux.npz", **pack(aux))
    atomic_json(directory / "witnesses.json", witnesses)
    atomic_json(directory / "type-keys.json", sorted(keys))
    manifest = dict(teacher=teacher, seed=seed, kind=kind, created_utc=now(),
                    phase_counts=dict(phase_counts), accounting=dict(stats),
                    optimization_targets_per_model=(config["native_budget"] + config["auxiliary_budget"]) if kind == "train" else 0,
                    distinct_complete_auxiliary_queries={k: len(v) for k, v in query_hashes.items()},
                    distinct_sufficient_types=len(keys),
                    generation_seconds=time.perf_counter()-tick, cpu_seconds=time.process_time()-cpu,
                    simulator_accounting="Observed states include filtering/unused phases. Native simulation and decomposition also compute unretained context; these are not claimed to be a supervised-label-only oracle cost.",
                    files={p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()})
    atomic_json(manifest_path, manifest)
    return manifest


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def objective(model, base, aux, bi, ai, arm, update, warmup):
    arrays = tuple(base[f"a{j}"][bi] for j in range(4))
    native = base["native"][bi]; phase = int(native[0, 3])
    out = model(*tensor_arrays(arrays), phase=phase)
    if phase == LINIAL:
        return F.cross_entropy(out.scores, torch.from_numpy(native[:, 1] // MAX_FIELD))
    if phase == DETECT:
        return F.binary_cross_entropy_with_logits(out.commit, torch.from_numpy(native[:, 2].astype(np.float32)))
    p, c = torch.from_numpy(base["p"][bi]), torch.from_numpy(base["c"][bi])
    loss_p = F.binary_cross_entropy_with_logits(out.state.p_logits[:, 0], p[:, 0].float())
    loss_c = F.cross_entropy(out.scores, c[:, 0])
    loss_native = F.binary_cross_entropy_with_logits(model.commit_for(out.state, p, c),
                                                    torch.from_numpy((native[:, 0] > 0).astype(np.float32)))
    if update < warmup:
        return loss_p + loss_c + loss_native
    aa = tuple(aux[f"a{j}"][ai] for j in range(4))
    state = model.first(tensor_arrays(aa), phase)
    prefix = "cf_" if arm == "cf" else ""
    pp, cc = torch.from_numpy(aux[prefix+"p"][ai]), torch.from_numpy(aux[prefix+"c"][ai])
    truth = torch.from_numpy(aux["cf_y" if arm == "cf" else "native_y"][ai])
    loss_aux = F.binary_cross_entropy_with_logits(model.commit_for(state, pp, cc), truth)
    return loss_p + loss_c + .5 * loss_native + .5 * loss_aux


def train(config, teacher, seed, arm, dataset, directory, frozen_sources):
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "record.json"
    if final.exists():
        rec = json.loads(final.read_text())
        if file_hash(directory / "final.pt") != rec["checkpoint_sha256"] or rec["sources"] != frozen_sources:
            raise RuntimeError("Frozen training mismatch")
        return rec
    base, aux = load_npz(dataset / "base.npz"), load_npz(dataset / "aux.npz")
    model = initialize(teacher, "normalized", seed); initial = weight_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    rng = np.random.default_rng(named_seed(config["namespace"], "order", seed))
    phases = sorted(np.unique(base["native"][:, 3]))
    populations = {int(p): np.flatnonzero(base["native"][:, 3] == p) for p in phases}
    apop = {int(p): np.flatnonzero(aux["native"][:, 3] == p) for p in phases}
    visits = np.zeros(len(base["native"]), np.int64); avisits = np.zeros(len(aux["native"]), np.int64)
    history = []; start = 0; elapsed = cpu_elapsed = 0.
    resume = directory / "last.pt"
    if resume.exists():
        old = torch.load(resume, weights_only=False)
        if old["sources"] != frozen_sources or old["config"] != config:
            raise RuntimeError("Resume sources/configuration mismatch")
        model.load_state_dict(old["model"]); optimizer.load_state_dict(old["optimizer"])
        rng.bit_generator.state = old["rng"]
        visits, avisits = old["visits"], old["avisits"]
        history, start, elapsed, cpu_elapsed = old["history"], old["update"], old["seconds"], old["cpu_seconds"]
    tick, cpu = time.perf_counter(), time.process_time()
    for update in range(start, config["updates"]):
        phase = int(phases[update % len(phases)])
        bi = rng.choice(populations[phase], config["batch_size"], replace=True)
        ai = rng.choice(apop[phase], config["batch_size"], replace=True) if len(apop[phase]) else np.array([], int)
        np.add.at(visits, bi, 1)
        if phase not in (LINIAL, DETECT) and update >= config["warmup_updates"]:
            np.add.at(avisits, ai, 1)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, base, aux, bi, ai, arm, update, config["warmup_updates"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward(); optimizer.step()
        if (update+1) % 100 == 0:
            history.append(dict(update=update+1, loss=float(loss.detach())))
        if (update+1) % config["save_every"] == 0:
            save_checkpoint(resume, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=rng.bit_generator.state, visits=visits, avisits=avisits, history=history,
                sources=frozen_sources, config=config, update=update+1,
                seconds=elapsed+time.perf_counter()-tick, cpu_seconds=cpu_elapsed+time.process_time()-cpu))
    record = dict(teacher=teacher, seed=seed, arm=arm, updates=config["updates"],
        initial_hash=initial, weights_hash=weight_hash(model), parameters=sum(p.numel() for p in model.parameters()),
        seconds=elapsed+time.perf_counter()-tick, cpu_seconds=cpu_elapsed+time.process_time()-cpu,
        base_presentations=int(visits.sum()), auxiliary_presentations=int(avisits.sum()),
        base_visited=int((visits>0).sum()), auxiliary_visited=int((avisits>0).sum()),
        visits_sha256=array_hash(visits, avisits), rng=rng.bit_generator.state,
        history=history, sources=frozen_sources, dataset_manifest_sha256=file_hash(dataset/"manifest.json"))
    save_checkpoint(directory/"final.pt", dict(model=model.state_dict(), record=record))
    record["checkpoint_sha256"] = file_hash(directory/"final.pt")
    atomic_json(final, record)
    print(json.dumps({k: record[k] for k in ("teacher", "seed", "arm", "seconds", "weights_hash")}), flush=True)
    return record


def confusion(pred, truth):
    tp, tn = int((pred & truth).sum()), int((~pred & ~truth).sum())
    fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, n=len(truth), accuracy=(tp+tn)/len(truth),
                balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None)


@torch.inference_mode()
def diagnostics(model, teacher, data, trainkeys):
    aux = load_npz(data/"aux.npz"); keys = json.loads((data/"type-keys.json").read_text())
    if set(keys) & trainkeys:
        raise RuntimeError("Train/test type leakage")
    pred = np.zeros(len(aux["native"]), bool); rowkeys = [None]*len(pred)
    for phase in sorted(np.unique(aux["native"][:, 3])):
        pos = np.flatnonzero(aux["native"][:, 3] == phase)
        for start in range(0, len(pos), 256):
            use = pos[start:start+256]
            arrays = tuple(aux[f"a{j}"][use] for j in range(4))
            state = model.first(tensor_arrays(arrays), int(phase))
            p, c = aux["cf_p"][use], aux["cf_c"][use]
            pred[use] = model.commit_for(state, torch.from_numpy(p), torch.from_numpy(c)).numpy() > 0
            for j, idx in enumerate(use):
                rowkeys[idx] = key_text(commit_key(arrays, j, p, c, teacher))
    truth = aux["cf_y"] > .5
    typed = defaultdict(list)
    for k, good in zip(rowkeys, pred == truth): typed[k].append(bool(good))
    result = confusion(pred, truth)
    result.update(distinct_types=len(typed), equal_type_accuracy=float(np.mean([np.mean(v) for v in typed.values()])),
                  train_test_type_intersection=0)
    x = aux["a0"][:, 0]; eligible = x[:, 7] > .5
    if teacher == "BE_BASE_D3": eligible &= x[:, 18] >= 0
    active = aux["cf_p"][:, 0]
    result["strata"] = {}
    for name, sel in (("eligible_active", eligible & active), ("eligible_silent", eligible & ~active),
                      ("ineligible", ~eligible)):
        if sel.any(): result["strata"][name] = confusion(pred[sel], truth[sel])
    return result


def final_test(config, runtime, frozen_sources):
    freeze_path = ROOT/"weights-frozen.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["sources"] != frozen_sources: raise RuntimeError("Sources changed after training")
    for rec in freeze["weights"]:
        if file_hash(rec["checkpoint"]) != rec["sha256"]: raise RuntimeError("Frozen weight changed")
    for teacher in config["teachers"]:
        for seed in config["seeds"]:
            out = ROOT/"prospective"/teacher.lower()/f"s{seed}"
            trainset = runtime/"datasets"/teacher.lower()/f"s{seed}"
            data = runtime/"test"/teacher.lower()/f"s{seed}"
            acquire(config, teacher, seed, data, "diagnostic")
            trainkeys = set(json.loads((trainset/"type-keys.json").read_text()))
            suite = cases(config, "final", seed)
            refpath = out/"teacher-reference.json"
            if not refpath.exists():
                reference, sec = evaluate(None, teacher, suite)
                summary = summarize_runs(reference, [m["case_id"] for _,m in suite])
                if summary["successes"] != len(suite): raise RuntimeError("Exact reference failed")
                atomic_json(refpath, dict(summary=summary, seconds=sec, rows=reference))
            for arm in ("native", "cf"):
                path = out/(arm+"-test.json")
                if path.exists(): continue
                model = initialize(teacher, "normalized", seed).eval()
                checkpoint = runtime/"models"/teacher.lower()/f"s{seed}"/arm/"final.pt"
                model.load_state_dict(torch.load(checkpoint, weights_only=False)["model"])
                diagnostic = diagnostics(model, teacher, data, trainkeys)
                rows, sec = evaluate(model, teacher, suite)
                summary = summarize_runs(rows, [m["case_id"] for _,m in suite])
                atomic_json(path, dict(teacher=teacher, seed=seed, arm=arm, diagnostic=diagnostic,
                    rollout_summary=summary, rollout_seconds=sec, rows=rows,
                    freeze_sha256=file_hash(freeze_path), test_manifest_sha256=file_hash(data/"manifest.json")))
                print(json.dumps(dict(stage="final",teacher=teacher,seed=seed,arm=arm,
                    diagnostic=diagnostic,rollouts=summary)), flush=True)


def run(stage, runtime):
    config = json.loads(CONFIG.read_text()); frozen_sources = sources()
    protocol = ROOT/"protocol-frozen.json"
    if not protocol.exists():
        atomic_json(protocol, dict(created_utc=now(), config=config, sources=frozen_sources,
            protocol_sha256=file_hash("docs/PAPER_REVISION_PROTOCOL.md"), final_test_opened=False))
    elif json.loads(protocol.read_text())["sources"] != frozen_sources:
        raise RuntimeError("Prospective source freeze differs")
    if stage in ("train", "all"):
        weights = []; pairing = []
        for teacher in config["teachers"]:
            for seed in config["seeds"]:
                data = runtime/"datasets"/teacher.lower()/f"s{seed}"
                manifest = acquire(config, teacher, seed, data)
                records = []
                for arm in ("native", "cf"):
                    directory = runtime/"models"/teacher.lower()/f"s{seed}"/arm
                    rec = train(config, teacher, seed, arm, data, directory, frozen_sources)
                    records.append(rec)
                    weights.append(dict(teacher=teacher,seed=seed,arm=arm,
                        checkpoint=str(directory/"final.pt"),sha256=rec["checkpoint_sha256"]))
                    atomic_json(ROOT/"training-status.json", dict(updated_utc=now(),complete=len(weights),total=40,last=weights[-1]))
                if records[0]["initial_hash"] != records[1]["initial_hash"] or records[0]["visits_sha256"] != records[1]["visits_sha256"]:
                    raise RuntimeError("Pair initialization/minibatches differ")
                states = [torch.load(runtime/"models"/teacher.lower()/f"s{seed}"/a/"final.pt",weights_only=False)["model"] for a in ("native","cf")]
                second = ("core.message2.","core.state2.","core.commit_head.","context_message2.","context_state2.")
                first_identical = all(torch.equal(states[0][k],states[1][k]) for k in states[0] if not k.startswith(second))
                if not first_identical: raise RuntimeError("Unexpected first-stage gradient difference")
                pairing.append(dict(teacher=teacher,seed=seed,initial_hash=records[0]["initial_hash"],
                    identical_minibatches=True,identical_first_stage_weights=True,targets_per_arm=manifest["optimization_targets_per_model"]))
        if not (ROOT/"weights-frozen.json").exists():
            atomic_json(ROOT/"weights-frozen.json",dict(created_utc=now(),sources=frozen_sources,weights=weights,pairing=pairing))
    if stage in ("test", "all"): final_test(config, runtime, frozen_sources)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("train","test","all"),required=True)
    parser.add_argument("--runtime",type=Path,required=True); args=parser.parse_args()
    run(args.stage,args.runtime)


if __name__ == "__main__": main()
