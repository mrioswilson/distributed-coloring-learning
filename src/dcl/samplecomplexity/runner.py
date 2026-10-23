"""Pilot -> prospective resource design -> exact validation -> freeze -> train."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from dcl.study.storage import atomic_json, file_hash
from .data import acquire, load
from .evaluation import evaluate
from .protocol import cases, digest, freeze, REGIME, source_hashes, summarize_runs, five_seed_gate
from .training import initialize, train

OUT = Path("results/sample-complexity")


def pilot(config, root):
    root = Path(root); OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); rows = []; data_rows = []
    for teacher in config["teachers"]:
        folder = root / "pilot" / "datasets" / teacher.lower()
        tick = time.perf_counter()
        manifest = acquire(config, teacher, 99101, config["pilot_query_budget"], folder, split="pilot")
        data_rows.append(dict(teacher=teacher, seconds=time.perf_counter()-tick, queries=manifest["queries"],
                              bytes=manifest["unique_stored_bytes"], uncompressed_bytes=manifest["uncompressed_bytes"]))
        for architecture in config["architectures"]:
            directory = root / "pilot" / "runs" / teacher.lower() / architecture
            record = train(config, teacher, architecture, 99101, config["pilot_query_budget"],
                           folder, directory, updates=config["pilot_updates"], dev=False)
            model = initialize(teacher, architecture, 99101)
            model.load_state_dict(torch.load(directory/"final.pt", weights_only=False)["model"])
            suite = cases(config, "pilot", 99101)
            runs, elapsed = evaluate(model, teacher, suite)
            atomic_json(directory / "rollouts.json", runs)
            rows.append(dict(teacher=teacher, architecture=architecture,
                             seconds_per_update=record["training_seconds"]/config["pilot_updates"],
                             seconds_per_execution=elapsed/len(suite), parameters=record["parameters"],
                             queries=record["accounting"]["queries"]))
    result = dict(config_sha256=digest(config), training=rows, data=data_rows, gpu_process_hours=0.,
                  wall_seconds=time.perf_counter()-started, device="cpu",
                  note="Timing pilot with independent namespace and initialization; outcomes never choose a teacher/architecture to omit.")
    atomic_json(OUT / "pilot.json", result)
    return result


def design(config, root):
    root = Path(root)
    p = json.loads((OUT / "pilot.json").read_text())
    if p["config_sha256"] != digest(config): raise RuntimeError("Pilot config changed")
    c = config; estimates = []
    for maximum in c["grid"]:
        grid = [b for b in c["grid"] if b <= maximum]
        train_s = sum(r["seconds_per_update"] for r in p["training"]) * c["updates"] * 5 * len(grid)
        dev_s = sum(r["seconds_per_execution"] for r in p["training"]) * c["development_runs"] * 5 * len(grid)
        data_s = sum(r["seconds"]/r["queries"] for r in p["data"]) * maximum * 5
        memory = max(r["uncompressed_bytes"]/r["queries"] for r in p["data"]) * maximum*c["workers"]
        disk = sum(r["bytes"]/r["queries"] for r in p["data"]) * maximum*5
        expected = c["cost_safety_factor"]*(train_s+dev_s+data_s)/c["workers"]
        estimates.append(dict(maximum=maximum, trainings=len(grid)*len(c["teachers"])*len(c["architectures"])*5,
                              supervisor_queries_acquired=maximum*len(c["teachers"])*5,
                              estimated_main_wall_seconds=expected,
                              estimated_total_seconds=expected+c["confirmation_reserve_seconds"]+c["controls_reserve_seconds"],
                              estimated_peak_dataset_ram_bytes=memory, estimated_dataset_disk_bytes=disk))
    feasible = [e for e in estimates if e["maximum"] <= c["main_grid_cap"] and
                e["estimated_total_seconds"] <= c["wall_budget_seconds"]]
    if not feasible: raise RuntimeError("Even the smallest balanced prefix exceeds the prospective budget")
    selected = feasible[-1]
    result = dict(config=config, grid=[b for b in c["grid"] if b <= selected["maximum"]],
                  selected=selected, estimates=estimates, pilot_sha256=file_hash(OUT/"pilot.json"),
                  selection="Largest timing-feasible common prefix, capped prospectively; identical cells for all seven teachers, three architectures and five seeds.",
                  source_hashes=source_hashes(),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  versions={k:version(k) for k in ("torch","numpy","networkx","scipy","matplotlib")},
                  python=platform.python_version(), device="cpu", gpu_process_hours=0.)
    freeze(OUT / "design.json", result); freeze(root / "design.json", result)
    return result


def validate_benchmark(config, root):
    """All prescribed tapes retained; a failed exact teacher aborts the freeze."""
    root = Path(root); manifests = []; summaries = []
    for split in ("development", "final"):
        for seed in config["seeds"]:
            suite = cases(config, split, seed)
            manifest = dict(split=split, seed=seed, cases=[m for _, m in suite])
            freeze(OUT / "batteries" / f"{split}-s{seed}.json", manifest)
            manifests.append(dict(split=split, seed=seed, sha256=digest(manifest)))
            for teacher in config["teachers"]:
                target = OUT / "exact" / teacher.lower() / f"{split}-s{seed}.json"
                if target.exists():
                    rows = json.loads(target.read_text()); elapsed = None
                else:
                    rows, elapsed = evaluate(None, teacher, suite)
                    atomic_json(target, rows)
                summary = summarize_runs(rows, [m["case_id"] for _, m in suite])
                summaries.append(dict(teacher=teacher, split=split, seed=seed, seconds=elapsed, **summary))
                if not summary["pass_all"]:
                    atomic_json(OUT / "exact-validation-failure.json", summaries)
                    raise RuntimeError(f"Exact teacher failed prospective battery: {teacher}/{split}/{seed}; no tape replacement allowed")
            print(f"exact validated {split}/s{seed}: {len(suite)} cases x 7 teachers", flush=True)
    result = dict(passed=True, batteries=manifests, exact=summaries,
                  disjoint_vertex_counts=dict(training=config["training_sizes"], development=config["development_sizes"], final=config["final_sizes"]),
                  note="Fresh graph/tape/ID seeds by split and training replica. Different vertex counts imply no graph overlap up to isomorphism across splits.")
    atomic_json(OUT / "exact-validation.json", result)
    frozen = dict(design_sha256=file_hash(OUT/"design.json"), config_sha256=digest(config),
                  source_hashes=source_hashes(), batteries=manifests,
                  exact_validation_sha256=file_hash(OUT/"exact-validation.json"))
    freeze(OUT / "preregistration.json", frozen)
    return result


def verify(config):
    frozen = json.loads((OUT/"preregistration.json").read_text())
    if frozen["config_sha256"] != digest(config) or frozen["source_hashes"] != source_hashes():
        raise RuntimeError("Frozen executable source/configuration changed; use a new protocol")
    if frozen["design_sha256"] != file_hash(OUT/"design.json"):
        raise RuntimeError("Design changed after preregistration")
    for b in frozen["batteries"]:
        path = OUT/"batteries"/f"{b['split']}-s{b['seed']}.json"
        if digest(json.loads(path.read_text())) != b["sha256"]: raise RuntimeError("Frozen evaluation battery changed")


def train_replica(args):
    config, root, teacher, seed, grid = args
    root = Path(root); torch.set_num_threads(1)
    folder = root / "datasets" / teacher.lower() / f"s{seed}"
    manifest = acquire(config, teacher, seed, max(grid), folder)
    atomic_json(OUT/"data"/teacher.lower()/f"s{seed}.json", manifest)
    records = []
    for budget in grid:
        prepared=load(folder,budget)
        for architecture in config["architectures"]:
            directory = root / "runs" / teacher.lower() / architecture / f"s{seed}" / f"b{budget}"
            r = train(config, teacher, architecture, seed, budget, folder, directory, prepared=prepared)
            atomic_json(OUT/"training"/teacher.lower()/architecture/f"s{seed}-b{budget}.json", r)
            records.append(r)
    return records


def cohort(config, root):
    verify(config)
    d = json.loads((OUT/"design.json").read_text()); jobs=[]
    # Interleave replicas across teachers for a balanced interruption boundary.
    for seed in config["seeds"]:
        for teacher in config["teachers"]:
            jobs.append((config, str(root), teacher, seed, d["grid"]))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(config["workers"], mp_context=context) as executor:
        results = list(executor.map(train_replica, jobs))
    atomic_json(OUT/"training.json", [r for cell in results for r in cell])


def confirm(config, root):
    verify(config); root = Path(root)
    records = json.loads((OUT/"training.json").read_text())
    grid = json.loads((OUT/"design.json").read_text())["grid"]
    candidates=[]
    for teacher in config["teachers"]:
        for architecture in config["architectures"]:
            match=[r for r in records if r["identity"]["teacher"]==teacher and r["identity"]["architecture"]==architecture]
            first = next((b for b in grid if len([r for r in match if r["identity"]["budget"]==b and r["development"]["pass_all"]])==5), None)
            checkpoints=[]
            if first is not None:
                for seed in config["seeds"]:
                    path=root/"runs"/teacher.lower()/architecture/f"s{seed}"/f"b{first}"/"final.pt"
                    checkpoints.append(dict(seed=seed, path=str(path), sha256=file_hash(path)))
            candidates.append(dict(teacher=teacher, architecture=architecture, budget=first, checkpoints=checkpoints))
    freeze(OUT/"frozen-candidates.json", dict(preregistration_sha256=file_hash(OUT/"preregistration.json"), candidates=candidates))
    results=[]
    for c in candidates:
        summaries={}
        consumption=OUT/"consumption"/f"{c['teacher'].lower()}-{c['architecture']}.json"
        if c["budget"] is not None:
            if consumption.exists():
                saved=json.loads(consumption.read_text())
                if saved.get("complete") is not True: raise RuntimeError("Final attempt already started; no automatic rerun permitted")
                results.append(saved["result"]); continue
            freeze(consumption, dict(complete=False, candidate=c))
            for cp in c["checkpoints"]:
                if file_hash(cp["path"]) != cp["sha256"]: raise RuntimeError("Frozen checkpoint changed")
                model=initialize(c["teacher"],c["architecture"],cp["seed"])
                model.load_state_dict(torch.load(cp["path"],weights_only=False)["model"])
                suite=cases(config,"final",cp["seed"])
                rows, elapsed=evaluate(model,c["teacher"],suite)
                target=OUT/"final"/c["teacher"].lower()/c["architecture"]/f"s{cp['seed']}.json"
                atomic_json(target, rows)
                summaries[cp["seed"]]=dict(**summarize_runs(rows,[m["case_id"] for _,m in suite]), seconds=elapsed, rows_sha256=file_hash(target))
                print(f"confirmation {c['teacher']}/{c['architecture']}/s{cp['seed']}: {summaries[cp['seed']]['successes']}/{len(suite)}", flush=True)
        gate=five_seed_gate(summaries, config["seeds"])
        result=dict(teacher=c["teacher"],architecture=c["architecture"],regime=REGIME[c["teacher"]],
                    candidate_budget=c["budget"], max_budget_evaluated=max(grid),
                    validated_budget=c["budget"] if gate["status"]=="PASS" else None,
                    **gate, per_seed=summaries,
                    final_attempted=c["budget"] is not None,
                    interpretation="Grid/development-selected candidate only; no final retry. Not an impossibility bound.")
        results.append(result)
        if c["budget"] is not None: atomic_json(consumption,dict(complete=True,result=result))
    atomic_json(OUT/"certification.json",results)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("pilot","design","validate","train","controls","confirm"))
    p.add_argument("--config",default="configs/sample-complexity.json")
    p.add_argument("--root",default="data/sample-complexity-v1")
    args=p.parse_args(); config=json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    from .controls import run as controls
    functions=dict(pilot=pilot,design=design,validate=validate_benchmark,train=cohort,controls=controls,confirm=confirm)
    functions[args.stage](config,args.root)


if __name__=="__main__": main()
