#!/usr/bin/env python3
"""Deterministic summaries and separate common-state/counterfactual diagnostics."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, DETECT, LINIAL, KW, MAX_FIELD, teacher_choices
from dcl.multiteacher_e1.model import native_prediction
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.full_fidelity import next_values
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash
from dcl.multicf.data import load
from dcl.multicf.oracles import confirmation, stage_targets, proposal_bank
from dcl.multicf.protocol import cases, horizon

OUT=Path("results/multiteacher-counterfactual")


def summaries():
    rows=[]
    for p in sorted((OUT/"runs").rglob("u*-record.json")):
        r=json.loads(p.read_text());i=r["identity"]
        rows.append(dict(**{k:i[k] for k in ("teacher","architecture","seed","method")},
            native_budget=int(p.parents[1].name.split("-")[0][1:]),native_labels=r["native_labels"],updates=r["update"],
            development_successes=r["development"]["successes"],development_executions=r["development"]["executions"],
            total_acquired_component_labels=r["total_acquired_component_labels"],
            counterfactual_labels=r["counterfactual_labels"],own_labels=r["own_confirmation_labels"],
            cpu_seconds=r["cpu_seconds"],training_seconds=r["training_seconds"],gpu_seconds=r["gpu_seconds"],
            parameters=r["parameters"],checkpoint=str(p).replace("-record.json",".pt"),
            checkpoint_sha256=r["checkpoint_sha256"],data_sha256=i["data_sha256"],initial_hash=r["initial_hash"]))
    if rows:
        with (OUT/"training-curves.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(OUT/"training-curves.json",rows)
    grouped=defaultdict(list)
    for r in rows:
        key=(r["teacher"],r["architecture"],r["method"],r["native_budget"],r["updates"])
        grouped[key].append(r)
    table=[]
    for key,rr in sorted(grouped.items()):
        teacher,arch,method,budget,updates=key
        table.append(dict(teacher=teacher,architecture=arch,method=method,native_labels=budget,updates=updates,
            training_seeds=len(rr),development_perfect_seeds=sum(r["development_successes"]==r["development_executions"] for r in rr),
            min_development_successes=min(r["development_successes"] for r in rr),
            acquired_labels_min=min(r["total_acquired_component_labels"] for r in rr),
            acquired_labels_max=max(r["total_acquired_component_labels"] for r in rr)))
    atomic_json(OUT/"development-summary.json",table)
    results=[json.loads(p.read_text()) for p in sorted((OUT/"confirmation").glob("*/result.json"))]
    atomic_json(OUT/"certification-summary.json",results)
    print(json.dumps(dict(checkpoints=len(rows),new_certifications=[(r["teacher"],r["architecture"],r["status"]) for r in results])),flush=True)
    return rows,table


def evaluation_bank(config,teacher,seed):
    import fcntl
    directory=OUT/"diagnostics"/"common-state-banks"/teacher.lower()/f"s{seed}"
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/".generation.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _evaluation_bank_unlocked(config,teacher,seed)


def _evaluation_bank_unlocked(config,teacher,seed):
    directory=OUT/"diagnostics"/"common-state-banks"/teacher.lower()/f"s{seed}"
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"states.npz"
    if path.exists():
        m=json.loads((directory/"manifest.json").read_text())
        if m["sha256"]!=file_hash(path):raise RuntimeError("Diagnostic bank hash changed")
        with np.load(path,allow_pickle=False) as raw:return {k:raw[k] for k in raw.files},m
    buffers={k:[] for k in ("a0","a1","a2","a3","native","meta")};transitions=[]
    for case_i,(g,case) in enumerate(cases(config,"development",seed)):
        p=ColoringProgram(g,teacher,case["tape_seed"],max_rounds=horizon(teacher,len(g)))
        while not p.finished:
            s=p.snapshot();exact=teacher_choices(s,p.neighbors,teacher);roots=np.flatnonzero(s.permission)
            for j,a in enumerate(encode_roots(s,p.neighbors,roots)):buffers[f"a{j}"].append(a)
            buffers["native"].append(np.c_[exact.color[roots],exact.auxiliary[roots],exact.mark[roots],np.full(len(roots),s.phase)])
            buffers["meta"].append(np.c_[s.colors[roots],s.aux[roots],np.full(len(roots),s.field),
                (s.bound-1-s.aux[roots])*4,np.full(len(roots),len(transitions)),np.full(len(roots),s.bound),
                np.full(len(roots),s.target),np.full(len(roots),case_i),np.full(len(roots),p.steps)])
            transitions.append(dict(case_id=case["case_id"],phase=s.phase,bound=s.bound,native_step=p.steps,roots=len(roots)))
            p.apply(s,exact)
        if p.failure:raise RuntimeError("Diagnostic exact teacher failed")
    packed={k:np.concatenate(v) for k,v in buffers.items()};np.savez_compressed(path,**packed)
    m=dict(teacher=teacher,seed=seed,namespace=config["namespace"],split="development",sha256=file_hash(path),
        transitions=transitions,scope="Evaluation references; no optimizer access")
    atomic_json(directory/"manifest.json",m)
    return packed,m


@torch.inference_mode()
def diagnostics(config,teacher,arch,seed,checkpoint,tag):
    torch.set_num_threads(1)
    packed,manifest=evaluation_bank(config,teacher,seed)
    model=initialize(teacher,arch,seed);model.load_state_dict(torch.load(checkpoint,weights_only=False)["model"]);model.eval()
    native,meta=packed["native"],packed["meta"]
    correct_action=np.zeros(len(native),bool);correct_state=np.zeros(len(native),bool)
    phase_counts=defaultdict(Counter);bound_counts=defaultdict(Counter);counts=Counter();errors=[]
    rng=np.random.default_rng(17001)
    for phase in sorted(np.unique(native[:,3])):
        positions=np.flatnonzero(native[:,3]==phase)
        for start in range(0,len(positions),256):
            use=positions[start:start+256]
            arrays=tuple(packed[f"a{i}"][use] for i in range(4))
            out=model(*tensor_arrays(arrays),phase=int(phase));predicted=tuple(t.numpy() for t in native_prediction(out))
            exact=tuple(native[use,j] for j in (0,1,2));field=1 if phase==LINIAL else (2 if phase==DETECT else 0)
            correct_action[use]=predicted[field]==exact[field]
            correct_state[use]=next_values(phase,predicted,meta[use])==next_values(phase,exact,meta[use])
            phase_counts[str(phase)]["action_total"]+=len(use)
            phase_counts[str(phase)]["action_correct"]+=int(correct_action[use].sum())
            for bound in np.unique(meta[use,5]):
                at=meta[use,5]==bound
                key=f"phase{phase}-bound{bound}"
                bound_counts[key]["action_total"]+=int(at.sum())
                bound_counts[key]["action_correct"]+=int(correct_action[use][at].sum())
            if phase not in (LINIAL,DETECT):
                targets=stage_targets(arrays,native[use],teacher)
                own_truth=confirmation(arrays,teacher,out.p.numpy(),out.candidates.numpy())
                counts["participation_total"]+=len(use)
                counts["participation_correct"]+=int((out.p[:,0].numpy()==targets.participate[:,0]).sum())
                counts["candidate_total"]+=len(use)
                counts["candidate_correct"]+=int((out.candidates[:,0].numpy()==targets.candidate[:,0]).sum())
                counts["own_confirmation_total"]+=len(use)
                counts["own_confirmation_correct"]+=int(((out.commit.numpy()>0)==(own_truth>.5)).sum())
                counts["unsafe_own_confirmation"]+=int(((out.commit.numpy()>0)&(own_truth<.5)&out.p[:,0].numpy()).sum())
                counts["own_false_negative"]+=int(((out.commit.numpy()<=0)&(own_truth>.5)).sum())
                ps,cs,ys=proposal_bank(arrays,native[use],teacher,rng)
                for v in range(4):
                    logits=model.commit_for(out.state,torch.from_numpy(ps[:,v]),torch.from_numpy(cs[:,v])).numpy()
                    counts["counterfactual_total"]+=len(use)
                    counts["counterfactual_correct"]+=int(((logits>0)==(ys[:,v]>.5)).sum())
                    counts["counterfactual_positive_truth"]+=int((ys[:,v]>.5).sum())
                    counts["counterfactual_false_positive"]+=int(((logits>0)&(ys[:,v]<.5)).sum())
                    counts["counterfactual_false_negative"]+=int(((logits<=0)&(ys[:,v]>.5)).sum())
            for j in np.flatnonzero(~correct_action[use])[:max(0,30-len(errors))]:
                errors.append(dict(phase=int(phase),bound=int(meta[use[j],5]),native_step=int(meta[use[j],8]),
                    case_id=manifest["transitions"][int(meta[use[j],4])]["case_id"],
                    predicted=int(predicted[field][j]),expected=int(exact[field][j])))
    mismatch=np.bincount(meta[:,4],weights=(~correct_state).astype(int),minlength=len(manifest["transitions"]))
    r=dict(teacher=teacher,architecture=arch,seed=seed,tag=tag,checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
        action_correct=int(correct_action.sum()),action_total=len(native),action_fidelity=float(correct_action.mean()),
        transition_equal=int((mismatch==0).sum()),transition_total=len(mismatch),
        transition_fidelity=float((mismatch==0).mean()),components=dict(counts),by_phase={k:dict(v) for k,v in phase_counts.items()},
        first_action_errors=errors,by_phase_bound={k:dict(v) for k,v in bound_counts.items()},bank_sha256=manifest["sha256"],
        scope="All permitted roots of 60 common native development trajectories; four fresh diagnostic continuation queries per color root; no training labels or final-case tuning")
    directory=OUT/"diagnostics"/tag/teacher.lower()/arch
    atomic_json(directory/f"s{seed}.json",r)
    return r


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=json.loads((OUT/"training-curves.json").read_text())
    teachers=[t for t in ("BE_BASE_D3","SPECULATIVE_GM","COLOR_TRIALS") if any(r["teacher"]==t for r in rows)]
    colors={"passive":"#718096","matched":"#d97706","counterfactual":"#007f86","owncf":"#9d174d"}
    fig,axes=plt.subplots(len(teachers),3,figsize=(11,3*len(teachers)),squeeze=False,sharey=True)
    for i,t in enumerate(teachers):
        for j,a in enumerate(("normalized","deepsets","transformer")):
            ax=axes[i,j]
            methods=sorted({r["method"] for r in rows if r["teacher"]==t and r["architecture"]==a})
            for m in methods:
                rr=[r for r in rows if r["teacher"]==t and r["architecture"]==a and r["method"]==m]
                for seed in sorted({r["seed"] for r in rr}):
                    ss=sorted([r for r in rr if r["seed"]==seed],key=lambda x:x["updates"])
                    ax.plot([r["updates"] for r in ss],[r["development_successes"]/r["development_executions"] for r in ss],
                        color=colors[m],alpha=.28,lw=.9)
                us=sorted({r["updates"] for r in rr})
                ax.plot(us,[np.mean([r["development_successes"]/r["development_executions"] for r in rr if r["updates"]==u]) for u in us],
                    color=colors[m],marker="o",label=m)
            ax.axhline(1,color="black",ls=":",lw=.8);ax.set_ylim(-.02,1.04);ax.set_xscale("log")
            ax.set_title(f"{t} / {a}",fontsize=10);ax.set_xlabel("Gradient updates")
            if j==0:ax.set_ylabel("Development success")
            ax.grid(alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=4)
    fig.tight_layout(rect=(0,.045,1,1));directory=OUT/"figures";directory.mkdir(exist_ok=True)
    for ext in ("pdf","png"):fig.savefig(directory/f"optimization-curves.{ext}",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["summarize","diagnostics","figures"])
    p.add_argument("--teacher");p.add_argument("--architecture");p.add_argument("--seed",type=int)
    p.add_argument("--checkpoint");p.add_argument("--tag",default="selected")
    p.add_argument("--config",default="configs/multiteacher-counterfactual.json");a=p.parse_args()
    if a.stage=="summarize":summaries()
    elif a.stage=="figures":figures()
    else:diagnostics(json.loads(Path(a.config).read_text()),a.teacher,a.architecture,a.seed,a.checkpoint,a.tag)
