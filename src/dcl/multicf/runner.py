"""Reproducible local campaign; at most three one-thread CPU workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import torch

from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash
from .data import acquire
from .protocol import cases, source_hashes, freeze, five_seed_gate, summarize_runs
from .training import train

ROOT = Path("results/multiteacher-counterfactual")


def task(job):
    config, teacher, architecture, seed, budget, method, coverage, alternate, updates, checkpoints, category = job
    torch.set_num_threads(1)
    suffix = f"b{budget}-coverage{int(coverage)}-alternate{int(alternate)}"
    dataset = ROOT / "datasets" / teacher.lower() / f"s{seed}" / suffix
    acquire(config, teacher, seed, budget, dataset, coverage, alternate)
    directory = ROOT / category / teacher.lower() / architecture / f"s{seed}" / suffix / method
    return train(config, teacher, architecture, seed, dataset, directory, method, updates, checkpoints)


def campaign(config, teachers, architectures, seeds, budget, methods, coverage, alternate, updates, checkpoints, workers=3, category="runs"):
    # Acquisition is deliberately sequential: three architecture jobs must
    # never race to create the same immutable bank.
    for teacher in teachers:
        for seed in seeds:
            acquire(config, teacher, seed, budget, ROOT/"datasets"/teacher.lower()/f"s{seed}"/
                    f"b{budget}-coverage{int(coverage)}-alternate{int(alternate)}", coverage, alternate)
    jobs = [(config,t,a,s,budget,m,coverage,alternate,updates,checkpoints,category)
        for t in teachers for a in architectures for s in seeds for m in methods]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(task, jobs))
    atomic_json(ROOT / f"{category}-index-{int(time.time())}.json", results)
    return results


def pilot(config):
    pc = dict(config, namespace=config["namespace"]+"-pilot", updates=24, checkpoints=[24],
              warmup_updates=4, development_runs=10, log_every=24, save_every=24)
    tick = time.perf_counter()
    result = campaign(pc, config["teachers"], config["architectures"], [99201], 600,
                      ["counterfactual"], True, True, 24, [24], 3, "pilot")
    records=[]
    for item in result:
        record=json.loads((Path(item["directory"])/"u000024-record.json").read_text())
        records.append(record)
    estimate = dict(optimization_seconds_per_4000_update_model={
        f"{r['identity']['teacher']}/{r['identity']['architecture']}":r["training_seconds"]*4000/24 for r in records},
        max_worker_rss_bytes=max(r["peak_rss_bytes"] for r in records), workers=3,
        gpu_seconds=0., pilot_wall_seconds=time.perf_counter()-tick,
        note="Linear pilot estimates; multiply optimization estimates by two for planning. Development and final evaluation are additional. No costly grid launched before this pilot.")
    atomic_json(ROOT/"pilot-resource-estimate.json",estimate)
    return estimate


def certify(config, teacher, architecture, budget, method, coverage=True, alternate=False, category="runs"):
    prefix = f"b{budget}-coverage{int(coverage)}-alternate{int(alternate)}"
    folders = {s:ROOT/category/teacher.lower()/architecture/f"s{s}"/prefix/method for s in config["seeds"]}
    candidate = None
    for u in config["checkpoints"]:
        paths = {s:d/f"u{u:06d}-record.json" for s,d in folders.items()}
        if all(p.exists() and json.loads(p.read_text()).get("development",{}).get("pass_all") is True for p in paths.values()):
            candidate=u;break
    name=f"{teacher.lower()}-{architecture}-{prefix}-{method}"
    out=ROOT/"confirmation"/name
    out.mkdir(parents=True,exist_ok=True)
    if candidate is None:
        record=dict(teacher=teacher,architecture=architecture,method=method,native_budget=budget,
                    status="NO DEVELOPMENT CANDIDATE",seeds_pass=None,final_attempted=False)
        atomic_json(out/"selection.json",record)
        return record
    frozen=[dict(seed=s,path=str(folders[s]/f"u{candidate:06d}.pt"),
        sha256=file_hash(folders[s]/f"u{candidate:06d}.pt")) for s in config["seeds"]]
    freeze(out/"freeze.json",dict(config=config,teacher=teacher,architecture=architecture,method=method,
        budget=budget,update=candidate,checkpoints=frozen,sources=source_hashes()))
    # Exact teachers validate every frozen tape before a student sees any final result.
    for seed in config["seeds"]:
        suite=cases(config,"final",seed)
        freeze(out/"batteries"/f"s{seed}.json",[m for _,m in suite])
        exact_path=ROOT/"exact-final"/teacher.lower()/f"s{seed}.json"
        if exact_path.exists():
            rows=json.loads(exact_path.read_text())
        else:
            rows,_=evaluate(None,teacher,suite);atomic_json(exact_path,rows)
        if not summarize_runs(rows,[m["case_id"] for _,m in suite])["pass_all"]:
            raise RuntimeError("Exact teacher failed; no replacement tape permitted")
    marker=out/"consumed.json"
    if marker.exists():
        previous=json.loads(marker.read_text())
        if previous.get("complete"):
            return json.loads((out/"result.json").read_text())
        raise RuntimeError("Final attempt already consumed; technical interruption must remain visible")
    freeze(marker,dict(complete=False,checkpoints=frozen))
    summaries={}
    for cp in frozen:
        if file_hash(cp["path"])!=cp["sha256"]:
            raise RuntimeError("Frozen checkpoint changed")
        model=initialize(teacher,architecture,cp["seed"])
        model.load_state_dict(torch.load(cp["path"],weights_only=False)["model"])
        suite=cases(config,"final",cp["seed"])
        rows,seconds=evaluate(model,teacher,suite)
        target=out/"final"/f"s{cp['seed']}.json";atomic_json(target,rows)
        summaries[cp["seed"]]=dict(**summarize_runs(rows,[m["case_id"] for _,m in suite]),
            seconds=seconds,rows_sha256=file_hash(target))
        print(json.dumps(dict(final=name,seed=cp["seed"],successes=summaries[cp["seed"]]["successes"])),flush=True)
    result=dict(teacher=teacher,architecture=architecture,method=method,native_budget=budget,update=candidate,
        **five_seed_gate(summaries,config["seeds"]),per_seed=summaries,final_attempted=True,
        interpretation="Separate new finite-battery certification; historical primary result remains unchanged")
    atomic_json(out/"result.json",result);atomic_json(marker,dict(complete=True,checkpoints=frozen))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["pilot","train","certify"])
    parser.add_argument("--config",default="configs/multiteacher-counterfactual.json")
    parser.add_argument("--teachers",nargs="+")
    parser.add_argument("--architectures",nargs="+")
    parser.add_argument("--seeds",nargs="+",type=int)
    parser.add_argument("--methods",nargs="+",default=["counterfactual"])
    parser.add_argument("--budget",type=int)
    parser.add_argument("--updates",type=int)
    parser.add_argument("--checkpoints",nargs="+",type=int)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--no-coverage",action="store_true")
    parser.add_argument("--alternate",action="store_true")
    parser.add_argument("--category",default="runs")
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    if args.stage=="pilot":
        pilot(config);return
    teachers=args.teachers or config["teachers"]
    architectures=args.architectures or config["architectures"]
    seeds=args.seeds or config["seeds"]
    budget=args.budget or config["native_budget"]
    updates=args.updates or config["updates"]
    checkpoints=args.checkpoints or config["checkpoints"]
    if args.stage=="train":
        campaign(config,teachers,architectures,seeds,budget,args.methods,not args.no_coverage,args.alternate,
                 updates,checkpoints,args.workers,args.category)
    else:
        for t in teachers:
            for a in architectures:
                for m in args.methods:
                    print(json.dumps(certify(config,t,a,budget,m,not args.no_coverage,args.alternate,args.category)))


if __name__=="__main__":
    main()
