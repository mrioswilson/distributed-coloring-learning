#!/usr/bin/env python3
"""Frozen-policy size transfer and explicit complete-output coupling bounds."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import beta
import torch

from dcl.multiteacher.programs import named_seed
from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json,file_hash
from dcl.multicf.protocol import cases,freeze,summarize_runs

OUT=Path("results/multiteacher-counterfactual")


def terminal(row):
    return tuple(row["colors"]),tuple(row[k] for k in
        ("proper","palette","invariants","terminated","no_technical_error","no_cutoff","terminal_correct"))


def suite_for(config,name,count):
    graph=nx.path_graph(8) if name=="path8" else nx.disjoint_union(nx.complete_graph(4),nx.empty_graph(4))
    return [(graph,dict(case_id=f"law-{name}-{i:04d}",n=8,family=name,
        tape_seed=named_seed(config["namespace"],"new-law-coupling",name,i))) for i in range(count)]


def transfer_suite(config,seed):
    suite=cases(config,"transfer",seed)
    for family in ("binary-tree","diamond-chain","triangle-chain","subdivided-K4"):
        for n in config["transfer_sizes"]:
            for replica in range(2):
                gs=named_seed(config["namespace"],"unseen-family",seed,family,n,replica)
                g=nx.Graph()
                if family=="binary-tree":
                    g.add_nodes_from(range(n))
                    g.add_edges_from(((v-1)//2,v) for v in range(1,n))
                elif family in ("diamond-chain","triangle-chain"):
                    width=4 if family=="diamond-chain" else 3
                    blocks=[]
                    for _ in range(n//width):
                        block=nx.complete_graph(width)
                        if width==4:block.remove_edge(0,1)
                        blocks.append(block)
                    g=nx.disjoint_union_all(blocks)
                    for i in range(len(blocks)-1):g.add_edge(i*width+1,(i+1)*width)
                    g.add_nodes_from(range(n))
                else:
                    component=nx.complete_graph(4)
                    for i,(u,v) in enumerate(list(component.edges())):
                        component.remove_edge(u,v);component.add_edges_from(((u,4+i),(4+i,v)))
                    g=nx.disjoint_union_all([component for _ in range(n//10)])
                    g.add_nodes_from(range(n))
                order=np.random.default_rng(gs).permutation(n)
                g=nx.relabel_nodes(g,dict(enumerate(map(int,order))))
                if max(dict(g.degree()).values())>3:raise RuntimeError("Transfer graph left native domain")
                suite.append((g,dict(case_id=f"unseen-{family}-n{n}-r{replica}-s{seed}",seed=seed,n=n,family=family,
                    edges=[list(e) for e in sorted(g.edges())],graph_seed=gs,
                    tape_seed=named_seed(config["namespace"],"unseen-tape",seed,family,n,replica))))
    return suite


def job(args):
    config,plan,selected,cp,stage=args;torch.set_num_threads(1)
    teacher,arch,seed=selected["teacher"],selected["architecture"],cp["seed"]
    model=initialize(teacher,arch,seed)
    if file_hash(cp["path"])!=cp["sha256"]:raise RuntimeError("Frozen policy changed")
    model.load_state_dict(torch.load(cp["path"],weights_only=False)["model"])
    directory=OUT/"generalization"/stage/teacher.lower()/arch/f"s{seed}"
    directory.mkdir(parents=True,exist_ok=True)
    if stage=="transfer":
        suite=transfer_suite(config,seed)
        rows,seconds=evaluate(model,teacher,suite)
        atomic_json(directory/"runs.json",rows)
        result=dict(teacher=teacher,architecture=arch,seed=seed,checkpoint_sha256=cp["sha256"],
            **summarize_runs(rows,[m["case_id"] for _,m in suite]),seconds=seconds,
            scope="Exploratory size/family transfer within degree<=3/global K4; no universal guarantee")
    else:
        results=[]
        for name in plan["graphs"]:
            suite=suite_for(config,name,plan["samples_per_fixed_graph"])
            reference_path=OUT/"generalization/law-references"/teacher.lower()/f"{name}.json"
            reference=json.loads(reference_path.read_text())
            rows,seconds=evaluate(model,teacher,suite)
            atomic_json(directory/f"{name}-runs.json",rows)
            disagreements=sum(terminal(a)!=terminal(b) for a,b in zip(reference,rows))
            n=len(rows);alpha=plan["family_alpha"]/plan["max_hypotheses"]
            upper=float(beta.ppf(1-alpha,disagreements+1,n-disagreements)) if disagreements<n else 1.
            freqs=Counter(json.dumps(terminal(r)) for r in rows)
            reference_freqs=Counter(json.dumps(terminal(r)) for r in reference)
            keys=set(freqs)|set(reference_freqs)
            empirical_tv=.5*sum(abs(freqs.get(k,0)/n-reference_freqs.get(k,0)/n) for k in keys)
            frequencies=[dict(outcome=json.loads(k),student_count=freqs.get(k,0),teacher_count=reference_freqs.get(k,0)) for k in sorted(keys)]
            atomic_json(directory/f"{name}-frequencies.json",frequencies)
            results.append(dict(graph=name,samples=n,coupled_disagreements=disagreements,
                tv_upper_simultaneous_95=upper,certifies_tv_at_most_005=upper<=.05,
                empirical_tv=empirical_tv,empirical_tv_scope="Descriptive same-coupling empirical histograms; not a population lower bound",
                student_mean_native_steps=float(np.mean([r["native_steps"] for r in rows])),
                teacher_mean_native_steps=float(np.mean([r["native_steps"] for r in reference])),
                student_summary=summarize_runs(rows,[m["case_id"] for _,m in suite]),seconds=seconds))
        result=dict(teacher=teacher,architecture=arch,seed=seed,checkpoint_sha256=cp["sha256"],graphs=results,
            scope="Fixed small-graph labelled terminal laws under independent per-case tapes/IDs. All status outcomes retained. Bonferroni family size 90; shared tapes between models do not create additional training replicas.")
    atomic_json(directory/"result.json",result)
    print(json.dumps(dict(supplement=stage,teacher=teacher,architecture=arch,seed=seed)),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["transfer","laws"])
    p.add_argument("--plan",default="initial-cohort");p.add_argument("--workers",type=int,default=3);a=p.parse_args()
    frozen=json.loads((OUT/"plans"/f"{a.plan}.json").read_text());config=frozen["config"]
    plan=json.loads(Path("configs/multiteacher-counterfactual-laws.json").read_text())
    selected=[]
    for s in frozen["selections"]:
        result_path=OUT/"confirmation"/s["name"]/"result.json"
        if s["method"]=="counterfactual" and result_path.exists() and json.loads(result_path.read_text())["status"]=="PASS":
            selected.append(s)
    freeze(OUT/"generalization"/f"{a.stage}-freeze.json",dict(plan=plan,selected=selected,
        namespace=config["namespace"],script_sha256=file_hash(__file__)))
    if a.stage=="laws":
        # Acquire each reference bank exactly once, before parallel policy jobs.
        for teacher in sorted({s["teacher"] for s in selected}):
            for name in plan["graphs"]:
                suite=suite_for(config,name,plan["samples_per_fixed_graph"])
                target=OUT/"generalization/law-references"/teacher.lower()/f"{name}.json"
                if not target.exists():rows,_=evaluate(None,teacher,suite);atomic_json(target,rows)
                rows=json.loads(target.read_text())
                if not summarize_runs(rows,[m["case_id"] for _,m in suite])["pass_all"]:
                    raise RuntimeError("Exact law reference failed; no discarded tape")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(job,[(config,plan,s,cp,a.stage) for s in selected for cp in s["checkpoints"]]))
    atomic_json(OUT/"generalization"/f"{a.stage}-summary.json",results)
