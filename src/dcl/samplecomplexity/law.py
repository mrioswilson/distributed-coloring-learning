"""Independent exploratory coloring-law banks using audited native-palette V2."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from dcl.integration.native_statistics import joint_test
from dcl.globaldist.core import holm
from dcl.multiteacher.programs import named_seed
from dcl.study.storage import atomic_json, file_hash
from .evaluation import evaluate
from .protocol import digest, freeze, graph_for, passed_run
from .training import initialize

OUT=Path("results/sample-complexity/law")


def bank(plan,teacher,label,graph,gid,model=None):
    path=OUT/"banks"/teacher.lower()/f"{gid}-{label}.npz"
    if path.exists(): return dict(np.load(path,allow_pickle=False))
    cases=[(graph,dict(case_id=f"{label}-{gid}-{i}",
                        tape_seed=named_seed(plan["namespace"],teacher,label,gid,i))) for i in range(plan["samples"])]
    rows,elapsed=evaluate(model,teacher,cases)
    arrays=dict(colors=np.asarray([r["colors"] for r in rows],np.int16),
                success=np.asarray([passed_run(r) for r in rows]),
                failure=np.asarray([r["failure"] or "" for r in rows]),
                steps=np.asarray([r["native_steps"] for r in rows],np.int16),
                observation_ids=np.arange(len(rows)))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f: np.savez_compressed(f,**arrays)
    atomic_json(path.with_suffix(".json"),dict(executions=len(rows),successes=int(arrays["success"].sum()),
                seconds=elapsed,sha256=file_hash(path),case_ids=[r["case_id"] for r in rows],
                note="All failures retained; successful rows used only for the conditional law test"))
    return arrays


def worker(args):
    main,plan,candidate,root=args
    torch.set_num_threads(1)
    t,a,b=candidate["teacher"],candidate["architecture"],candidate["budget"]
    rows=[]
    for seed in main["seeds"]:
        checkpoint=Path(root)/"runs"/t.lower()/a/f"s{seed}"/f"b{b}"/"final.pt"
        model=initialize(t,a,seed)
        model.load_state_dict(torch.load(checkpoint,weights_only=False)["model"])
        for gid,family,n in plan["graphs"]:
            target=OUT/"tests"/f"{t}-{a}-s{seed}-{gid}.json"
            if target.exists(): rows.append(json.loads(target.read_text()));continue
            graph=graph_for(family,n,named_seed(plan["namespace"],gid))
            reference=dict(np.load(OUT/"banks"/t.lower()/f"{gid}-exact.npz",allow_pickle=False))
            label=f"{a}-s{seed}"
            learned=bank(plan,t,label,graph,gid,model)
            x,y=reference["colors"][reference["success"]],learned["colors"][learned["success"]]
            result=dict(teacher=t,architecture=a,budget=b,seed=seed,graph=gid,
                        valid_teacher=len(x),valid_model=len(y),samples_per_bank=plan["samples"],
                        checkpoint_sha256=file_hash(checkpoint),
                        teacher_bank_sha256=file_hash(OUT/"banks"/t.lower()/f"{gid}-exact.npz"),
                        model_bank_sha256=file_hash(OUT/"banks"/t.lower()/f"{gid}-{label}.npz"))
            if min(len(x),len(y))<4:
                result.update(testable=False,p=1.,reason="Fewer than four successful rows; no equality inference")
            else:
                edges=np.asarray(sorted(graph.edges()),int).reshape(-1,2)
                meta=dict(id=gid,n=n,degrees=[graph.degree(i) for i in range(n)],
                          edges=np.concatenate((edges,edges[:,::-1])).T.tolist(),palette_size=4)
                stats=joint_test(x,y,meta,np.random.default_rng(named_seed(plan["namespace"],t,a,seed,gid,"permutations")),
                                 B=plan["permutations"],futility=plan["futility_exceedances_all_components"],
                                 x_ids=reference["observation_ids"][reference["success"]],
                                 y_ids=learned["observation_ids"][learned["success"]])
                result.update(testable=True,p=stats["p"],statistics=stats["statistics"],
                              permutation_count=stats["permutations_used"],p_kind=stats["p_kind"],
                              selection_per_class=stats["selection_per_class"],evaluation_per_class=stats["evaluation_per_class"])
            atomic_json(target,result);rows.append(result)
    print(f"law banks complete {t}/{a}",flush=True)
    return rows


def run(root="data/sample-complexity-v1"):
    main=json.loads(Path("configs/sample-complexity.json").read_text())
    plan=json.loads(Path("configs/sample-complexity-law.json").read_text())
    records=json.loads(Path("results/sample-complexity/training.json").read_text())
    grid=json.loads(Path("results/sample-complexity/design.json").read_text())["grid"]
    candidates=[]
    for t in main["teachers"]:
        for a in main["architectures"]:
            rs=[r for r in records if r["identity"]["teacher"]==t and r["identity"]["architecture"]==a]
            b=next((b for b in grid if sum(r["development"]["pass_all"] for r in rs if r["identity"]["budget"]==b)==5),max(grid))
            candidates.append(dict(teacher=t,architecture=a,budget=b))
    freeze(OUT/"frozen-plan.json",dict(plan=plan,selection=candidates,main_preregistration_sha256=file_hash("results/sample-complexity/preregistration.json"),
               source_hashes={str(p.relative_to(Path.cwd()) if p.is_absolute() else p):file_hash(p) for p in
                              [Path(__file__),Path("src/dcl/integration/native_statistics.py"),Path("src/dcl/robustdist/stats.py")]},
               note="Separate exploratory supplement; independent of and unable to select final certification candidates"))
    torch.set_num_threads(1)
    for t in main["teachers"]:
        for gid,family,n in plan["graphs"]:
            bank(plan,t,"exact",graph_for(family,n,named_seed(plan["namespace"],gid)),gid)
    with ProcessPoolExecutor(plan["workers"],mp_context=multiprocessing.get_context("spawn")) as executor:
        groups=list(executor.map(worker,[(main,plan,c,root) for c in candidates]))
    rows=[r for group in groups for r in group]
    if len(rows)!=210: raise RuntimeError("Incomplete planned law family")
    # holm returns Boolean decisions; adjusted p-values are computed explicitly.
    p=np.asarray([r["p"] for r in rows]);order=np.argsort(p,kind="stable")
    adjusted=np.empty(len(p));adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    for r,adj in zip(rows,adjusted): r.update(p_holm=float(adj),reject=bool(adj<=.05))
    atomic_json(OUT/"results.json",dict(planned=210,testable=sum(r["testable"] for r in rows),
                rejections=sum(r["reject"] for r in rows),rows=rows,
                scope="Conditional complete-coloring laws, fixed small graphs, one frozen checkpoint per replica; no equality inference from nonrejection"))


if __name__=="__main__": run()
