#!/usr/bin/env python3
"""Expanded development screening, cohort freeze, and once-only confirmation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json,file_hash
from dcl.multicf.protocol import cases,source_hashes,freeze,summarize_runs,five_seed_gate

OUT=Path("results/multiteacher-counterfactual")


def screen(job):
    config,teacher,arch,budget,method,category=job
    torch.set_num_threads(1)
    amendment=json.loads(Path("configs/multiteacher-counterfactual-screen.json").read_text())
    screening=dict(config,development_runs=amendment["development_runs"])
    suffix=f"b{budget}-coverage1-alternate0"
    name=f"{teacher.lower()}-{arch}-{suffix}-{method}"
    directory=OUT/"screening"/name
    freeze(directory/"protocol.json",dict(amendment=amendment,config=config,teacher=teacher,
        architecture=arch,budget=budget,method=method,script_sha256=file_hash(__file__)))
    selected=None;attempts=[]
    for u in config["checkpoints"]:
        folders={s:OUT/category/teacher.lower()/arch/f"s{s}"/suffix/method for s in config["seeds"]}
        records={s:d/f"u{u:06d}-record.json" for s,d in folders.items()}
        if not all(p.exists() for p in records.values()):
            continue
        if not all(json.loads(p.read_text()).get("development",{}).get("pass_all") for p in records.values()):
            attempts.append(dict(update=u,eligible_initial_development=False));continue
        summaries={};checkpoints=[]
        for s,d in folders.items():
            checkpoint=d/f"u{u:06d}.pt"
            cp=dict(seed=s,path=str(checkpoint),sha256=file_hash(checkpoint))
            checkpoints.append(cp)
            suite=cases(screening,"development",s)
            target=directory/f"u{u:06d}"/f"s{s}.json"
            freeze(target.with_suffix(".identity.json"),dict(checkpoint=cp,
                cases=[m for _,m in suite],namespace=config["namespace"]))
            if target.exists():
                rows=json.loads(target.read_text())
            else:
                model=initialize(teacher,arch,s)
                model.load_state_dict(torch.load(checkpoint,weights_only=False)["model"])
                rows,_=evaluate(model,teacher,suite)
                atomic_json(target,rows)
            summaries[s]=summarize_runs(rows,[m["case_id"] for _,m in suite])
        all_pass=all(s["pass_all"] for s in summaries.values())
        attempts.append(dict(update=u,eligible_initial_development=True,all_five_pass=all_pass,per_seed=summaries))
        if all_pass:
            selected=dict(teacher=teacher,architecture=arch,budget=budget,method=method,update=u,
                          name=name,checkpoints=checkpoints,screening_cases_per_seed=screening["development_runs"])
            break
    record=dict(name=name,teacher=teacher,architecture=arch,budget=budget,method=method,
        selected=selected,attempts=attempts,status="CANDIDATE" if selected else "NO DEVELOPMENT CANDIDATE")
    atomic_json(directory/"selection.json",record)
    print(json.dumps(dict(screen=name,update=selected["update"] if selected else None)),flush=True)
    return record


def confirm(job):
    config,selected=job
    torch.set_num_threads(1)
    teacher,arch=selected["teacher"],selected["architecture"]
    directory=OUT/"confirmation"/selected["name"]
    directory.mkdir(parents=True,exist_ok=True)
    freeze(directory/"freeze.json",dict(config=config,selected=selected,sources=source_hashes(),
        confirmation_script_sha256=file_hash(__file__)))
    for cp in selected["checkpoints"]:
        if file_hash(cp["path"])!=cp["sha256"]:raise RuntimeError("Frozen checkpoint changed")
    # Keep exact validation beside each package, avoiding shared-writer races.
    for s in config["seeds"]:
        suite=cases(config,"final",s)
        freeze(directory/"batteries"/f"s{s}.json",[m for _,m in suite])
        target=directory/"exact"/f"s{s}.json"
        if target.exists():rows=json.loads(target.read_text())
        else:rows,_=evaluate(None,teacher,suite);atomic_json(target,rows)
        if not summarize_runs(rows,[m["case_id"] for _,m in suite])["pass_all"]:
            raise RuntimeError("Exact teacher failed a frozen final tape; cannot replace it")
    marker=directory/"consumed.json"
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get("complete"):return json.loads((directory/"result.json").read_text())
        raise RuntimeError("Final cohort attempt already consumed; no automatic retry")
    freeze(marker,dict(complete=False,checkpoints=selected["checkpoints"]))
    summaries={}
    for cp in selected["checkpoints"]:
        s=cp["seed"];model=initialize(teacher,arch,s)
        model.load_state_dict(torch.load(cp["path"],weights_only=False)["model"])
        suite=cases(config,"final",s)
        rows,seconds=evaluate(model,teacher,suite)
        target=directory/"final"/f"s{s}.json";atomic_json(target,rows)
        summaries[s]=dict(**summarize_runs(rows,[m["case_id"] for _,m in suite]),seconds=seconds,
                          rows_sha256=file_hash(target))
        print(json.dumps(dict(final=selected["name"],seed=s,successes=summaries[s]["successes"])),flush=True)
    result=dict(teacher=teacher,architecture=arch,method=selected["method"],native_budget=selected["budget"],
        update=selected["update"],**five_seed_gate(summaries,config["seeds"]),per_seed=summaries,
        final_attempted=True,scope="New finite-battery certification after expanded development; historical primary result unchanged")
    atomic_json(directory/"result.json",result);atomic_json(marker,dict(complete=True,checkpoints=selected["checkpoints"]))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["screen","freeze","run"])
    p.add_argument("--config",default="configs/multiteacher-counterfactual.json")
    p.add_argument("--teachers",nargs="+",default=["BE_BASE_D3","SPECULATIVE_GM","COLOR_TRIALS"])
    p.add_argument("--methods",nargs="+",default=["counterfactual","matched","passive"])
    p.add_argument("--budget",type=int);p.add_argument("--workers",type=int,default=3)
    p.add_argument("--plan",default="initial-cohort");p.add_argument("--category",default="runs")
    args=p.parse_args();config=json.loads(Path(args.config).read_text());torch.set_num_threads(1)
    if args.stage=="screen":
        jobs=[(config,t,a,args.budget or (30000 if t=="COLOR_TRIALS" else 3000),m,args.category)
              for t in args.teachers for a in config["architectures"] for m in args.methods]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:records=list(pool.map(screen,jobs))
        atomic_json(OUT/f"screening-{args.plan}.json",records)
    elif args.stage=="freeze":
        selections=[]
        for r in json.loads((OUT/f"screening-{args.plan}.json").read_text()):
            if r["selected"]:selections.append(r["selected"])
        freeze(OUT/"plans"/f"{args.plan}.json",dict(config=config,selections=selections,sources=source_hashes(),
            script_sha256=file_hash(__file__),screening_sha256=file_hash(OUT/f"screening-{args.plan}.json")))
        print(json.dumps(dict(frozen_packages=len(selections))),flush=True)
    else:
        plan=json.loads((OUT/"plans"/f"{args.plan}.json").read_text())
        if plan["sources"]!=source_hashes() or plan["script_sha256"]!=file_hash(__file__):raise RuntimeError("Frozen source changed")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(confirm,[(plan["config"],s) for s in plan["selections"]]))
        atomic_json(OUT/f"confirmation-{args.plan}.json",results)
