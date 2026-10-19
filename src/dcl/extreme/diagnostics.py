"""Controlled local states; observations never feed back into checkpoints.

The cached state is the actual first learned communication at the center of
a star. Only r_c changes during a sweep, so recomputing neighbors for each
draw would be redundant. The exact teacher is used exclusively for scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .common import atomic_json, digest, now, provenance, weights_hash
from .runtime import ChunkedPolicy
from dcl.universal.train import load_model


@torch.inference_mode()
def learned_center(model, colors, palette):
    device=next(model.parameters()).device
    colors=torch.as_tensor(colors,dtype=torch.long,device=device)
    slots=torch.arange(palette,device=device)
    received=torch.zeros((palette,model.message1[0].out_features),device=device)
    # Each message is learned from the original input, with no available-set mask.
    for start in range(1,len(colors),32):
        c=colors[start:start+32]
        own=(c[:,None]==slots).float()
        f=torch.stack(((c<0).float()[:,None].expand_as(own),own),-1)
        received=torch.maximum(received,model.message1(f).amax(0))
    h=model.state1(torch.cat((torch.ones((palette,1),device=device),received),-1))
    alpha=torch.sigmoid(model.availability(h).squeeze(-1))
    khat=alpha.sum()
    prefix=alpha.cumsum(0)-alpha
    lo,hi=prefix/khat.clamp_min(.5),(prefix+alpha)/khat.clamp_min(.5)
    return h,alpha,khat,prefix,lo,hi


@torch.inference_mode()
def choices_from_center(model, state, grid):
    h,alpha,khat,prefix,lo,hi=state
    device=h.device
    selections=[]
    for start in range(0,len(grid),128):
        r=torch.as_tensor(grid[start:start+128],dtype=torch.float32,device=device)
        g,k=len(r),len(alpha)
        broadcast=lambda x:x[None,:].expand(g,k)
        extra=(broadcast(alpha),broadcast(lo),broadcast(hi),r[:,None].expand(g,k)) if model.variant=="NORMALIZED" else (
            broadcast(alpha),broadcast(prefix),broadcast(prefix+alpha),khat.expand(g,k),r[:,None].expand(g,k))
        scores=model.scorer(torch.cat((h[None,:,:].expand(g,k,-1),torch.stack(extra,-1)),-1)).squeeze(-1)
        selections.append(scores.argmax(1).cpu().numpy())
    return np.concatenate(selections)


def rc_grid(k):
    # Float32 inputs are the policy's actual numeric interface. Compare the
    # exact teacher at those representable values, not at a hidden float64 rc.
    grid=((np.arange(1001)+.5)/1001).astype(np.float32)
    boundaries=(np.arange(1,k,dtype=np.float64)/k).astype(np.float32)
    near=np.concatenate((np.nextafter(boundaries,np.float32(0)),boundaries,
                         np.nextafter(boundaries,np.float32(1)),[np.float32(0),np.nextafter(np.float32(1),np.float32(0))]))
    return np.unique(np.concatenate((grid,near))).astype(np.float32),boundaries


def selection_rows(model, directory, palettes=(4,11,21,31,51,101,201,501)):
    rows=[]
    directory=Path(directory)
    for palette in palettes:
        for k in sorted({v for v in (1,2,3,4,5,10,20,50,100,200,500,palette) if v<=palette}):
            for pattern in ("LOW_AVAILABLE","HIGH_AVAILABLE","SCATTERED"):
                rng=np.random.default_rng(int(digest([palette,k,pattern])[:16],16))
                available=(np.arange(k) if pattern=="LOW_AVAILABLE" else np.arange(palette-k,palette)
                           if pattern=="HIGH_AVAILABLE" else np.sort(rng.choice(palette,k,replace=False)))
                occupied=np.setdiff1d(np.arange(palette),available)
                colors=np.full(palette,-1,np.int64)
                colors[1:1+len(occupied)]=occupied
                state=learned_center(model,colors,palette)
                grid,bounds=rc_grid(k)
                choices=choices_from_center(model,state,grid)
                # The star is a valid partial coloring; neighbors have no edges
                # between them. The central node has degree palette-1.
                ranks=np.minimum(np.floor(grid.astype(np.float64)*k).astype(int),k-1)
                teacher=available[ranks]
                forbidden=~np.isin(choices,available)
                alpha=state[1].cpu().numpy()
                relative=alpha/max(float(alpha.max()),1e-12)
                exact=np.isin(np.arange(palette),available)
                prefix=exact.cumsum()-exact
                lo,hi=state[4].cpu().numpy(),state[5].cpu().numpy()
                bulk=np.isin(grid,((np.arange(1001)+.5)/1001).astype(np.float32))
                boundary_mask=~bulk
                row={"palette":palette,"degree":palette-1,"k":k,"pattern":pattern,"draws":len(grid),
                     "candidate_correct":int((choices==teacher).sum()),"forbidden":int(forbidden.sum()),
                     "uniform_grid_accuracy":float((choices[bulk]==teacher[bulk]).mean()),
                     "boundary_draws":int(boundary_mask.sum()),
                     "boundary_correct":int(((choices==teacher)&boundary_mask).sum()),
                     "availability_slot_errors":int(((relative>=.5)!=exact).sum()),
                     "k_hat_raw":float(state[2]),"relative_count_estimate":float(relative.sum()),
                     "max_interval_endpoint_error":float(np.maximum(abs(lo-prefix/k),abs(hi-(prefix+exact)/k)).max()),
                     "distinct_proposed_colors":int(len(np.unique(choices))),
                     "selected_rank_monotone":bool(np.all(np.diff(choices)>=0))}
                path=directory/f"K{palette}-k{k}-{pattern}.npz"
                directory.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary,rc=grid,choices=choices,teacher=teacher,available=available,
                                    alpha=alpha,lo=lo,hi=hi,colors=colors)
                temporary.replace(path)
                rows.append(row)
    return rows


@torch.inference_mode()
def commit_rows(model,palettes=(4,11,21,31,51,101,201,501)):
    device=next(model.parameters()).device
    policy=ChunkedPolicy(model)
    rows=[]
    for palette in palettes:
        degree=palette-1
        # Color identities include endpoints and interior; semantics are held fixed.
        color_names=sorted({0,1,palette//2,palette-2,palette-1})
        for color in color_names:
            other=(color+1)%palette
            for kind in ("VALID","FIXED","SIMULTANEOUS","MULTIPLE_FIXED","MULTIPLE_SIMULTANEOUS","BOTH","WAIT"):
                n=degree+1
                colors=np.full(n,-1,np.int64)
                part=np.zeros(n,bool)
                cand=np.full(n,other,np.int64)
                part[0]=kind!="WAIT"
                cand[0]=color
                if kind in ("FIXED","MULTIPLE_FIXED","BOTH"):
                    colors[1:1+(min(degree,3) if kind=="MULTIPLE_FIXED" else 1)]=color
                if kind in ("SIMULTANEOUS","MULTIPLE_SIMULTANEOUS","BOTH"):
                    begin=2 if kind=="BOTH" else 1
                    end=min(n,begin+(3 if kind=="MULTIPLE_SIMULTANEOUS" else 1))
                    part[begin:end]=True
                    cand[begin:end]=color
                edges=np.array([(0,j) for j in range(1,n)]+[(j,0) for j in range(1,n)]).T
                args=(torch.tensor(colors,device=device),torch.tensor(edges,device=device),
                      torch.full((n,),palette,dtype=torch.long,device=device),
                      torch.tensor(part,device=device),torch.tensor(cand,device=device))
                logit=float(policy.commit_logits(*args)[0])
                valid=kind=="VALID"
                predicted=logit>0
                rows.append({"palette":palette,"degree":degree,"color":color,"kind":kind,
                             "valid":valid,"predicted_commit":predicted,"logit":logit,
                             "unsafe_commit":bool(part[0] and predicted and not valid),
                             "valid_rejected":bool(valid and not predicted)})
    return rows


@torch.inference_mode()
def commit_presence_table(model):
    """All 64 sets of feasible neighbor message types at the proposed color.

    At a central uncolored, participating node with an in-palette candidate,
    its slot input is [1,0,1,1]. A neighbor has exactly six feasible binary
    types. Max aggregation depends only on which types occur, not their
    multiplicities. This characterizes this COMMIT pathway, not the complete
    coloring algorithm or candidate rule.
    """
    device=next(model.parameters()).device
    types=torch.tensor([[0,0,0,0],[0,1,0,0],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1]],dtype=torch.float32,device=device)
    messages=model.message2(types)
    center=torch.tensor([1,0,1,1],dtype=torch.float32,device=device)
    rows=[]
    for mask in range(64):
        indices=[i for i in range(6) if mask&(1<<i)]
        received=messages[indices].amax(0) if indices else torch.zeros_like(messages[0])
        h=model.state2(torch.cat((center,received)))[None,:]
        head=torch.cat((h,torch.ones((1,3),device=device)),-1)
        logit=float(model.commit_head(head).squeeze())
        valid=not (mask&(1<<1) or mask&(1<<5))
        rows.append({"neighbor_type_mask":mask,"types_present":len(indices),"valid":valid,
                     "predicted_commit":logit>0,"logit":logit,"correct":bool((logit>0)==valid),
                     "unsafe_commit":bool(logit>0 and not valid)})
    return rows


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp.csv")
    with temporary.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [])
        w.writeheader()
        w.writerows(rows)
    temporary.replace(path)


def run_model(model_info,root,device):
    root=Path(root)
    directory=root/"diagnostics"/device/model_info["run"]
    done=directory/"done.json"
    if done.exists():
        saved_report=json.loads(done.read_text())
        if saved_report.get("diagnostic_version")==2 and saved_report["weights_sha256_before"]==model_info["weights_sha256"]:
            return saved_report
    model,saved=load_model(model_info["checkpoint"],device)
    before=weights_hash(model)
    if before!=model_info["weights_sha256"]: raise RuntimeError("Wrong diagnostic checkpoint")
    selection=selection_rows(model,directory/"selection-raw")
    commit=commit_rows(model)
    presence=commit_presence_table(model)
    write_csv(directory/"selection.csv",selection)
    write_csv(directory/"commit.csv",commit)
    write_csv(directory/"commit-presence-exhaustive.csv",presence)
    # Participation is independent of edges and palette in this architecture.
    rp=np.unique(np.concatenate((np.linspace(0,1,10001,dtype=np.float32),
                                 [np.nextafter(np.float32(.5),np.float32(0)),np.float32(.5),
                                  np.nextafter(np.float32(.5),np.float32(1))])))
    with torch.inference_mode():
        r=torch.tensor(rp,device=device)
        logits=model.participation(torch.stack((r,torch.ones_like(r)),-1)).squeeze(-1).cpu().numpy()
    part={"draws":len(rp),"correct":int(((logits>0)==(rp<.5)).sum()),
          "largest_participating_rp":float(rp[logits>0].max()) if (logits>0).any() else None}
    assert weights_hash(model)==before
    report={"diagnostic_version":2,"checkpoint":model_info["checkpoint"],"weights_sha256_before":before,"weights_sha256_after":weights_hash(model),
            "device":device,"selection_states":len(selection),"commit_states":len(commit),
            "unsafe_counterfactual_commits":sum(r["unsafe_commit"] for r in commit),"participation":part,
            "commit_presence_cases":len(presence),"commit_presence_correct":sum(r["correct"] for r in presence),
            "commit_presence_min_abs_logit":min(abs(r["logit"]) for r in presence),
            "provenance":provenance(),"completed_utc":now()}
    atomic_json(done,report)
    print(f"diagnostic {model_info['run']} states={len(selection)} unsafe={report['unsafe_counterfactual_commits']}",flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--run",help="Exact inventory run name; otherwise all E1 and seed-0 controls")
    args=p.parse_args()
    torch.set_num_threads(1)
    inventory=json.loads((Path(args.root)/"audit/inventory.json").read_text())
    for r in inventory["runs"]:
        if not r.get("checkpoint"): continue
        selected=(r["run"]==args.run) if args.run else (r["config"]["name"]=="E1" or r["config"]["seed"]==0)
        if selected: run_model(r,args.root,args.device)


if __name__=="__main__": main()
