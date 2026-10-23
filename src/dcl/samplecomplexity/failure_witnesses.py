"""Post hoc structural witnesses; frozen models are never adjusted or repaired."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ASSIGN, ColoringProgram, Choices, named_seed
from dcl.multiteacher_e1.model import native_prediction
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.study.storage import atomic_json, file_hash
from .data import components
from .evaluation import evaluate
from .protocol import cases, horizon
from .training import initialize

OUT=Path("results/sample-complexity")


@torch.inference_mode()
def run(root="data/sample-complexity-v1"):
    config=json.loads(Path("configs/sample-complexity.json").read_text());root=Path(root)
    cert=json.loads((OUT/"certification.json").read_text());rows=[];summaries=[]
    torch.set_num_threads(1)
    for arch in config["architectures"]:
        failures=[]
        for seed in config["seeds"]:
            path=OUT/"final/speculative_gm"/arch/f"s{seed}.json"
            if not path.exists():continue
            for row in json.loads(path.read_text()):
                if row["failure"] is None:continue
                degree=np.zeros(row["n"],int)
                for u,v in row["edges"]:degree[u]+=1;degree[v]+=1
                pending=np.flatnonzero(np.asarray(row["colors"])<0)
                failures.append(dict(case_id=row["case_id"],seed=seed,family=row["family"],n=row["n"],
                        failure=row["failure"],pending_vertices=pending.tolist(),
                        pending_degrees=degree[pending].tolist(),all_pending_isolated=bool(len(pending) and (degree[pending]==0).all())))
        summaries.append(dict(architecture=arch,failures=failures,failed_cases=len(failures),
                              all_pending_isolated=bool(failures) and all(r["all_pending_isolated"] for r in failures)))
    for seed in config["seeds"]:
        folder=root/"datasets/speculative_gm"/f"s{seed}"
        with np.load(folder/"dataset.npz",allow_pickle=False) as stored:
            all_labels=stored["labels"];all_x=stored["a0"]
        cumulative=np.cumsum([len(components(int(p))) for p in all_labels[:,0]])
        for arch in config["architectures"]:
            c=next(c for c in cert if c["teacher"]=="SPECULATIVE_GM" and c["architecture"]==arch)
            budget=c["candidate_budget"]
            if budget is None:continue
            take=np.searchsorted(cumulative,budget,side="right");labels=all_labels[:take]
            isolated=all_x[:take,0,39]==0
            populations={int(p):np.flatnonzero(labels[:,0]==p) for p in np.unique(labels[:,0])};phases=sorted(populations)
            order=np.random.default_rng(named_seed(config["namespace"],"batch-order",seed));visits=np.zeros(take,np.int64)
            for u in range(config["updates"]):
                positions=order.choice(populations[phases[u%len(phases)]],config["batch_size"],replace=True)
                np.add.at(visits,positions,1)
            record=json.loads((root/"runs/speculative_gm"/arch/f"s{seed}"/f"b{budget}"/"record.json").read_text())
            assert hashlib.sha256(visits.tobytes()).hexdigest()==record["per_example_visit_hash"]
            acquired=(labels[:,0]==ASSIGN)&isolated
            checkpoint=root/"runs/speculative_gm"/arch/f"s{seed}"/f"b{budget}"/"final.pt"
            model=initialize("SPECULATIVE_GM",arch,seed)
            model.load_state_dict(torch.load(checkpoint,weights_only=False)["model"]);model.eval()
            tape=named_seed("posthoc-empty-neighborhood-witness",seed)
            p=ColoringProgram(nx.empty_graph(1),"SPECULATIVE_GM",tape,max_rounds=horizon("SPECULATIVE_GM",1));trace=[]
            while not p.finished:
                state=p.snapshot();roots=np.flatnonzero(state.permission)
                out=model(*tensor_arrays(encode_roots(state,p.neighbors,roots)),phase=state.phase)
                color,aux,mark=[a.numpy() for a in native_prediction(out)]
                choice=Choices(color.astype(int),aux.astype(int),mark)
                trace.append(dict(phase=state.phase,before=p.colors.tolist(),participation=out.p[:,0].tolist(),
                       participation_logit=out.state.p_logits[:,0].tolist(),commit_logit=out.commit.tolist(),
                       native_color_action_code=color.tolist(),learned_candidate_color=out.candidates[:,0].tolist(),mark=mark.tolist()))
                p.apply(state,choice)
            exact,_=evaluate(None,"SPECULATIVE_GM",[(nx.empty_graph(1),dict(case_id="singleton",tape_seed=tape))])
            dev=cases(config,"development",seed)
            rows.append(dict(teacher="SPECULATIVE_GM",architecture=arch,seed=seed,budget=budget,
                singleton_success=p.failure is None,singleton_failure=p.failure,trace=trace,
                exact_singleton_success=exact[0]["terminal_correct"],checkpoint_sha256=file_hash(checkpoint),
                acquired_isolated_assign_bundles=int(acquired.sum()),
                presented_isolated_assign_bundles=int(visits[acquired].sum()),
                distinct_isolated_assign_bundles_presented=int((visits[acquired]>0).sum()),
                isolated_vertices_in_development_suite=sum(sum(d==0 for _,d in g.degree()) for g,_ in dev)))
    atomic_json(OUT/"structural-failure-witnesses.json",dict(final_failure_summary=summaries,singleton_witnesses=rows,
         scope="Exploratory diagnostics of frozen candidate weights, selected after observing final failures. No checkpoint/label/optimizer changes and no repeated final run. Singleton is a new diagnostic graph, not a certification battery.",
         caveat="Shared failures across architectures may reflect shared core initialization/training. Acquired/presented counts diagnose coverage and fit; they do not prove a universal optimization or capacity theorem.",source_sha256=file_hash(__file__)))


if __name__=="__main__":run()
