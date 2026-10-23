"""Paired independent confirmation of the fixed-label optimizer control."""
import json
from pathlib import Path

import torch

from dcl.study.storage import atomic_json, file_hash
from .evaluation import evaluate
from .protocol import cases, five_seed_gate, freeze, summarize_runs, passed_run
from .training import initialize

OUT=Path("results/sample-complexity/optimization-confirmation")


def run(root="data/sample-complexity-v1"):
    config=json.loads(Path("configs/sample-complexity.json").read_text())
    plan=json.loads(Path("configs/sample-complexity-optimization.json").read_text())
    final_config=dict(config,namespace=plan["namespace"]);root=Path(root)
    frozen=[]
    for teacher in plan["teachers"]:
        for updates in plan["updates"]:
            for seed in config["seeds"]:
                path=(root/"runs"/teacher.lower()/"normalized"/f"s{seed}"/f"b{plan['budget']}"/"final.pt" if updates==500 else
                      root/"controls/optimization"/teacher.lower()/f"s{seed}"/"final.pt")
                frozen.append(dict(teacher=teacher,updates=updates,seed=seed,path=str(path),sha256=file_hash(path)))
    matched=[]
    for teacher in plan["teachers"]:
        for seed in config["seeds"]:
            pair=[next(c for c in frozen if (c["teacher"],c["seed"],c["updates"])==(teacher,seed,u)) for u in plan["updates"]]
            a,b=[torch.load(c["path"],weights_only=False)["record"] for c in pair]
            if a["initial_hash"]!=b["initial_hash"] or a["accounting"]["prefix_sha256"]!=b["accounting"]["prefix_sha256"]:
                raise RuntimeError("Optimizer arms must have identical initialization and supervised labels")
            matched.append(dict(teacher=teacher,seed=seed,initial_hash=a["initial_hash"],
                    prefix_sha256=a["accounting"]["prefix_sha256"],actual_queries=a["accounting"]["queries"]))
    freeze(OUT/"frozen-plan.json",dict(plan=plan,checkpoints=frozen,source_sha256=file_hash(__file__),
           matched_labels_and_initialization=matched,
           main_preregistration_sha256=file_hash("results/sample-complexity/preregistration.json")))
    torch.set_num_threads(1)
    for seed in config["seeds"]:
        suite=cases(final_config,"final",seed)
        freeze(OUT/"batteries"/f"s{seed}.json",dict(seed=seed,cases=[m for _,m in suite]))
        for teacher in plan["teachers"]:
            path=OUT/"exact"/teacher.lower()/f"s{seed}.json"
            if path.exists():rows=json.loads(path.read_text())
            else:rows,_=evaluate(None,teacher,suite);atomic_json(path,rows)
            if not summarize_runs(rows,[m["case_id"] for _,m in suite])["pass_all"]:
                raise RuntimeError("Optimizer-control exact teacher failed new frozen battery; no tape replacement")
    results=[]
    for teacher in plan["teachers"]:
        marker=OUT/"consumption"/f"{teacher}.json"
        if marker.exists():
            old=json.loads(marker.read_text())
            if not old.get("complete"):raise RuntimeError("Paired optimizer-control final attempt already consumed")
            results.extend(old["results"]);continue
        freeze(marker,dict(complete=False,teacher=teacher))
        arms=[];raw_by_arm={}
        for updates in plan["updates"]:
            summaries={};raw_by_arm[updates]={}
            for seed in config["seeds"]:
                cp=next(c for c in frozen if (c["teacher"],c["updates"],c["seed"])==(teacher,updates,seed))
                if file_hash(cp["path"])!=cp["sha256"]:raise RuntimeError("Frozen optimizer-control checkpoint changed")
                model=initialize(teacher,"normalized",seed)
                model.load_state_dict(torch.load(cp["path"],weights_only=False)["model"])
                suite=cases(final_config,"final",seed)
                rows,elapsed=evaluate(model,teacher,suite)
                target=OUT/"final"/teacher.lower()/f"u{updates}"/f"s{seed}.json";atomic_json(target,rows)
                summaries[seed]=dict(**summarize_runs(rows,[m["case_id"] for _,m in suite]),seconds=elapsed,rows_sha256=file_hash(target))
                raw_by_arm[updates][seed]=rows
                print(f"fixed-label confirmation {teacher}/u{updates}/s{seed}: {summaries[seed]['successes']}/1000",flush=True)
            gate=five_seed_gate(summaries,config["seeds"])
            arms.append(dict(teacher=teacher,architecture="normalized",budget=plan["budget"],updates=updates,**gate,per_seed=summaries,
                             interpretation="Validated fixed point, not a minimum over optimizer or label budgets"))
        paired=[]
        for seed in config["seeds"]:
            a,b=raw_by_arm[500][seed],raw_by_arm[1500][seed]
            if [r["case_id"] for r in a]!=[r["case_id"] for r in b]:raise RuntimeError("Paired control tapes differ")
            paired.append(dict(seed=seed,both_success=sum(passed_run(x) and passed_run(y) for x,y in zip(a,b)),
                   only_1500_success=sum(not passed_run(x) and passed_run(y) for x,y in zip(a,b)),
                   only_500_success=sum(passed_run(x) and not passed_run(y) for x,y in zip(a,b)),
                   neither_success=sum(not passed_run(x) and not passed_run(y) for x,y in zip(a,b))))
        for arm in arms:arm["paired_seed_outcomes"]=paired
        atomic_json(marker,dict(complete=True,results=arms));results.extend(arms)
    atomic_json(OUT/"results.json",dict(arms=results,student_executions=30000,exact_executions=15000,
       scope=plan["scope"],note="Paired evaluation is within each fixed policy pair; executions are not independent training replicas"))


if __name__=="__main__":run()
