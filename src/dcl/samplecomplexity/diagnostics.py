"""Exploratory common-state phase/component diagnostics of difficult teachers."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, DETECT, KW, LINIAL, named_seed, teacher_choices
from dcl.multiteacher_e1.model import native_prediction
from dcl.multiteacher_e1.supervision import labels_from_inputs, tensor_arrays
from dcl.study.storage import atomic_json, file_hash
from .protocol import cases, horizon
from .training import initialize

OUT=Path("results/sample-complexity")


@torch.inference_mode()
def run(root="data/sample-complexity-v1"):
    config=json.loads(Path("configs/sample-complexity.json").read_text());root=Path(root)
    grid=json.loads((OUT/"design.json").read_text())["grid"];budget=max(grid)
    rows=[];failures=[];torch.set_num_threads(1)
    for teacher in ("JP","BE_BASE_D3","SPECULATIVE_GM"):
        for seed in config["seeds"]:
            batches=[[],[],[],[]];labs=[];categories=[]
            for graph,meta in cases(config,"development",seed):
                p=ColoringProgram(graph,teacher,meta["tape_seed"],max_rounds=horizon(teacher,len(graph)))
                while not p.finished:
                    state=p.snapshot();exact=teacher_choices(state,p.neighbors,teacher)
                    roots=np.flatnonzero(state.permission);arrays=encode_roots(state,p.neighbors,roots)
                    native=np.column_stack((exact.color[roots],exact.auxiliary[roots],exact.mark[roots],np.full(len(roots),state.phase)))
                    labels=np.zeros((len(roots),7),int);labels[:,0]=state.phase
                    labels[:,4:]=native[:,:3]
                    if state.phase not in (LINIAL,DETECT):
                        targets=labels_from_inputs(arrays,native,teacher)
                        labels[:,1]=targets.participate[:,0];labels[:,2]=targets.candidate[:,0];labels[:,3]=exact.color[roots]>0
                    for b,a in zip(batches,arrays): b.append(a)
                    labs.append(labels)
                    categories.extend(np.sign(state.aux[roots]-state.target).astype(int).tolist() if state.phase==KW else [9]*len(roots))
                    p.apply(state,exact)
            arrays=tuple(np.concatenate(b) for b in batches);labels=np.concatenate(labs);categories=np.asarray(categories)
            for architecture in config["architectures"]:
                checkpoint=root/"runs"/teacher.lower()/architecture/f"s{seed}"/f"b{budget}"/"final.pt"
                model=initialize(teacher,architecture,seed)
                model.load_state_dict(torch.load(checkpoint,weights_only=False)["model"]);model.eval()
                counts=defaultdict(lambda:[0,0])
                for phase in sorted(np.unique(labels[:,0])):
                    indices=np.flatnonzero(labels[:,0]==phase)
                    for start in range(0,len(indices),256):
                        use=indices[start:start+256];truth=labels[use]
                        out=model(*tensor_arrays(tuple(a[use] for a in arrays)),phase=int(phase))
                        pred=native_prediction(out);field=1 if phase==LINIAL else (2 if phase==DETECT else 0)
                        measures={"native_action":pred[field].numpy()==truth[:,4+field]}
                        if phase not in (LINIAL,DETECT):
                            measures.update(participation=out.p[:,0].numpy()==truth[:,1],
                                            proposal=out.candidates[:,0].numpy()==truth[:,2],
                                            teacher_confirmation=(out.commit.numpy()>0)==truth[:,3])
                        for name,equal in measures.items():
                            key=(int(phase),name,"all");counts[key][0]+=int(equal.sum());counts[key][1]+=len(equal)
                            if phase==KW:
                                for category in (-1,0,1):
                                    chosen=categories[use]==category
                                    key=(int(phase),name,f"aux_minus_target_sign_{category}")
                                    counts[key][0]+=int(equal[chosen].sum());counts[key][1]+=int(chosen.sum())
                for (phase,component,category),(correct,total) in counts.items():
                    rows.append(dict(teacher=teacher,architecture=architecture,seed=seed,budget=budget,phase=phase,
                          component=component,category=category,correct=correct,total=total,
                          accuracy=correct/total if total else None,checkpoint_sha256=file_hash(checkpoint)))
                for b in grid:
                    path=root/"runs"/teacher.lower()/architecture/f"s{seed}"/f"b{b}"/"development.json"
                    raw=json.loads(path.read_text());counts_failure=Counter()
                    for row in raw:
                        if row["failure"]:
                            stage=("Linial" if row["native_steps"]<=2 else "KW") if teacher=="BE_BASE_D3" else "native"
                            counts_failure[stage,row["failure"]]+=1
                    failures.append(dict(teacher=teacher,architecture=architecture,seed=seed,budget=b,
                        first_failure_phase=[dict(phase=p,failure=f,count=n) for (p,f),n in counts_failure.items()],
                        source_sha256=file_hash(path)))
    atomic_json(OUT/"difficult-teacher-diagnostics.json",dict(rows=rows,autonomous_failures=failures,
        selection="JP/BE/GM, all three main architectures, all five seeds, maximum main budget, every development teacher state; no final data or optimizer",
        interpretation="Component agreement on teacher states and first autonomous failure phase localize errors. They do not identify data/optimization/representation as the sole cause.",
        evaluation_labels_are_not_training_supervision=True,source_sha256=file_hash(__file__)))


if __name__=="__main__":run()
