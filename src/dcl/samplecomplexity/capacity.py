"""Small-model supervision curves, independent supplemental confirmation."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import torch

from dcl.study.storage import atomic_json, file_hash
from .data import load
from .evaluation import evaluate
from .protocol import cases, five_seed_gate, freeze, summarize_runs
from .training import initialize, train

OUT=Path("results/sample-complexity/capacity")


def worker(args):
    config,plan,root,teacher,seed=args
    root=Path(root);dataset=root/"datasets"/teacher.lower()/f"s{seed}"
    rows=[]
    for budget in plan["grid"]:
        directory=(root/"controls"/"capacity"/teacher.lower()/"small"/f"s{seed}" if budget==config["control_budget"] else
                   root/"capacity-grid"/teacher.lower()/f"s{seed}"/f"b{budget}")
        r=train(config,teacher,"small",seed,budget,dataset,directory,prepared=load(dataset,budget))
        r=dict(r,checkpoint=str(directory/"final.pt"));rows.append(r)
        atomic_json(OUT/"training"/teacher.lower()/f"s{seed}-b{budget}.json",r)
    return rows


def run(root="data/sample-complexity-v1"):
    config=json.loads(Path("configs/sample-complexity.json").read_text())
    plan=json.loads(Path("configs/sample-complexity-capacity.json").read_text())
    freeze(OUT/"plan.json",dict(plan=plan,main_preregistration_sha256=file_hash("results/sample-complexity/preregistration.json"),
                  source_sha256=file_hash(__file__),scope="Exploratory small-grid supplement, frozen before any small-model final results"))
    jobs=[(config,plan,root,t,s) for s in config["seeds"] for t in plan["teachers"]]
    torch.set_num_threads(1)
    with ProcessPoolExecutor(plan["workers"],mp_context=multiprocessing.get_context("spawn")) as executor:
        result=list(executor.map(worker,jobs))
    records=[r for group in result for r in group];atomic_json(OUT/"training.json",records)
    final_config=dict(config,namespace=plan["namespace"])
    candidates=[]
    for t in plan["teachers"]:
        first=next((b for b in plan["grid"] if sum(r["development"]["pass_all"] for r in records if
                       r["identity"]["teacher"]==t and r["identity"]["budget"]==b)==5),None)
        checkpoints=[dict(seed=r["identity"]["seed"],path=r["checkpoint"],sha256=r["checkpoint_sha256"]) for r in records if
                     r["identity"]["teacher"]==t and r["identity"]["budget"]==first]
        candidates.append(dict(teacher=t,budget=first,checkpoints=checkpoints))
    freeze(OUT/"frozen-candidates.json",dict(candidates=candidates))
    for s in config["seeds"]:
        suite=cases(final_config,"final",s)
        freeze(OUT/"batteries"/f"s{s}.json",dict(seed=s,cases=[m for _,m in suite]))
        for t in plan["teachers"]:
            path=OUT/"exact"/t.lower()/f"s{s}.json"
            if path.exists(): rows=json.loads(path.read_text())
            else:
                rows,_=evaluate(None,t,suite);atomic_json(path,rows)
            if not summarize_runs(rows,[m["case_id"] for _,m in suite])["pass_all"]:
                raise RuntimeError("Small-study exact teacher failed independent frozen final battery; no tape replacement")
    certifications=[]
    for c in candidates:
        summaries={};marker=OUT/"consumption"/f"{c['teacher']}.json"
        if c["budget"] is not None:
            if marker.exists():
                saved=json.loads(marker.read_text())
                if not saved.get("complete"): raise RuntimeError("Small-model final attempt already consumed")
                certifications.append(saved["result"]);continue
            freeze(marker,dict(complete=False,candidate=c))
            for cp in c["checkpoints"]:
                if file_hash(cp["path"])!=cp["sha256"]: raise RuntimeError("Small checkpoint changed")
                model=initialize(c["teacher"],"small",cp["seed"])
                model.load_state_dict(torch.load(cp["path"],weights_only=False)["model"])
                suite=cases(final_config,"final",cp["seed"])
                rows,elapsed=evaluate(model,c["teacher"],suite)
                path=OUT/"final"/c["teacher"].lower()/f"s{cp['seed']}.json";atomic_json(path,rows)
                summaries[cp["seed"]]=dict(**summarize_runs(rows,[m["case_id"] for _,m in suite]),seconds=elapsed,rows_sha256=file_hash(path))
        gate=five_seed_gate(summaries,config["seeds"])
        result=dict(teacher=c["teacher"],architecture="small",candidate_budget=c["budget"],
                    validated_budget=c["budget"] if gate["status"]=="PASS" else None,**gate,per_seed=summaries,
                    final_attempted=c["budget"] is not None,parameters=next(r["parameters"] for r in records if r["identity"]["teacher"]==c["teacher"]))
        certifications.append(result)
        if c["budget"] is not None: atomic_json(marker,dict(complete=True,result=result))
    atomic_json(OUT/"certification.json",certifications)


if __name__=="__main__": run()
