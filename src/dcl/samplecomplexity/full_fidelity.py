"""Complete development teacher-state action and actual-transition audit."""
import json
from pathlib import Path

import numpy as np
import torch

from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, DETECT, LINIAL, KW, MAX_FIELD, teacher_choices
from dcl.multiteacher_e1.model import native_prediction
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.study.storage import atomic_json, file_hash
from .protocol import cases, horizon
from .training import initialize

OUT=Path("results/sample-complexity")


def next_values(phase,actions,meta):
    """Primary updated state channel; equal channels imply equal interpreter state.

    Redundant GM repair / KW writes can differ as actions but leave state equal.
    All phase/scheduler/RNG updates are then identical from this common state.
    """
    color,aux,mark=actions
    if phase==LINIAL:return (aux//MAX_FIELD)*meta[:,2]+aux%MAX_FIELD
    if phase==KW:return np.where(color>0,meta[:,3]+color-1,meta[:,1])
    if phase==DETECT:return mark.astype(int)
    return np.where(color>0,color-1,meta[:,0])


def bank(config,teacher,seed,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/"teacher-states.npz"
    if path.exists():
        manifest=json.loads((folder/"manifest.json").read_text())
        if file_hash(path)!=manifest["sha256"]:raise RuntimeError("Full-fidelity teacher bank changed")
        with np.load(path,allow_pickle=False) as saved:return {k:saved[k] for k in saved.files},manifest
    buffers=[[],[],[],[]];labels=[];metadata=[];transitions=[]
    for case_i,(graph,record) in enumerate(cases(config,"development",seed)):
        p=ColoringProgram(graph,teacher,record["tape_seed"],max_rounds=horizon(teacher,len(graph)))
        while not p.finished:
            state=p.snapshot();choices=teacher_choices(state,p.neighbors,teacher);roots=np.flatnonzero(state.permission)
            arrays=encode_roots(state,p.neighbors,roots)
            for b,a in zip(buffers,arrays):b.append(a)
            labels.append(np.column_stack((np.full(len(roots),state.phase),choices.color[roots],
                                            choices.auxiliary[roots],choices.mark[roots])))
            metadata.append(np.column_stack((state.colors[roots],state.aux[roots],np.full(len(roots),state.field),
                              (state.bound-1-state.aux[roots])*4,np.full(len(roots),len(transitions)),
                              np.full(len(roots),state.bound),np.full(len(roots),state.target),
                              np.full(len(roots),case_i),np.full(len(roots),p.steps))))
            transitions.append(dict(case_id=record["case_id"],native_step=p.steps,phase=state.phase,bound=state.bound,roots=len(roots)))
            p.apply(state,choices)
        if p.failure:raise RuntimeError("Exact development teacher failed; no replacement permitted")
    arrays={f"a{i}":np.concatenate(b) for i,b in enumerate(buffers)}
    arrays.update(labels=np.concatenate(labels).astype(np.int64),meta=np.concatenate(metadata).astype(np.int64))
    np.savez_compressed(path,**arrays)
    manifest=dict(teacher=teacher,seed=seed,split="development",sha256=file_hash(path),
                  native_action_labels=len(arrays["labels"]),transitions=transitions,
                  scope="Evaluation references only, never training supervision; all 28 frozen development trajectories")
    atomic_json(folder/"manifest.json",manifest)
    return arrays,manifest


@torch.inference_mode()
def run(root="data/sample-complexity-v1"):
    root=Path(root);config=json.loads(Path("configs/sample-complexity.json").read_text())
    training=json.loads((OUT/"training.json").read_text());rows=[];banks=[]
    torch.set_num_threads(1)
    for teacher in config["teachers"]:
        for seed in config["seeds"]:
            data,manifest=bank(config,teacher,seed,root/"full-fidelity"/teacher.lower()/f"s{seed}")
            banks.append(dict(teacher=teacher,seed=seed,sha256=manifest["sha256"],native_action_labels=manifest["native_action_labels"],transitions=len(manifest["transitions"])))
            labels,meta=data["labels"],data["meta"];arrays=tuple(data[f"a{i}"] for i in range(4))
            fields=np.where(labels[:,0]==LINIAL,meta[:,2],4)
            groups=sorted(set(zip(labels[:,0].tolist(),fields.tolist())))
            for record in training:
                identity=record["identity"]
                if (identity["teacher"],identity["seed"])!=(teacher,seed):continue
                checkpoint=root/"runs"/teacher.lower()/identity["architecture"]/f"s{seed}"/f"b{identity['budget']}"/"final.pt"
                if file_hash(checkpoint)!=record["checkpoint_sha256"]:raise RuntimeError("Frozen full-fidelity checkpoint changed")
                model=initialize(teacher,identity["architecture"],seed)
                model.load_state_dict(torch.load(checkpoint,weights_only=False)["model"]);model.eval()
                correct_action=np.zeros(len(labels),bool);correct_state=np.zeros(len(labels),bool)
                for phase,field in groups:
                    positions=np.flatnonzero((labels[:,0]==phase)&(fields==field))
                    for start in range(0,len(positions),256):
                        use=positions[start:start+256]
                        out=model(*tensor_arrays(tuple(a[use] for a in arrays)),phase=int(phase))
                        predicted=tuple(t.numpy() for t in native_prediction(out))
                        exact=tuple(labels[use,j] for j in (1,2,3))
                        index=1 if phase==LINIAL else (2 if phase==DETECT else 0)
                        correct_action[use]=predicted[index]==exact[index]
                        correct_state[use]=next_values(phase,predicted,meta[use])==next_values(phase,exact,meta[use])
                mismatches=np.bincount(meta[:,4],weights=(~correct_state).astype(int),minlength=len(manifest["transitions"]))
                phases=[]
                for phase,bound in sorted({(int(p),int(b)) for p,b in zip(labels[:,0],meta[:,5])}):
                    use=(labels[:,0]==phase)&(meta[:,5]==bound)
                    phases.append(dict(phase=phase,bound=bound,total=int(use.sum()),action_correct=int(correct_action[use].sum()),
                                       state_channel_correct=int(correct_state[use].sum())))
                rows.append(dict(**identity,action_total=len(labels),action_correct=int(correct_action.sum()),
                     action_fidelity=float(correct_action.mean()),transitions=len(mismatches),
                     identical_transitions=int((mismatches==0).sum()),transition_fidelity=float((mismatches==0).mean()),
                     by_phase_and_bound=phases,checkpoint_sha256=record["checkpoint_sha256"],teacher_bank_sha256=manifest["sha256"]))
            print(f"full common-state fidelity {teacher}/s{seed}",flush=True)
    atomic_json(OUT/"full-fidelity.json",dict(rows=rows,banks=banks,source_sha256=file_hash(__file__),
             scope="Post hoc descriptive audit on every root of every full development teacher trajectory; no final data, training updates, candidate changes or new certification",
             transition_definition="Equality of actual updated primary state channels and hence interpreter state from each common teacher state; redundant GM repair/KW actions may agree in state while disagreeing in action",
             original_prefix_metrics="Training records retain the earlier 256-decision prefix and action-identical transition proxy; full audit is reported separately"))


if __name__=="__main__":run()
