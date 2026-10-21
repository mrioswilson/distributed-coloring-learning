"""Frozen candidate predictions, finite safety checks and controlled progress.

All exact availability operations below are observers/state constructors.
Models receive only their original interface. No parameters are updated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from dcl.study.storage import atomic_json,file_hash,now
from dcl.universal.model import mlp
from .certification import TYPES,commit_certificate,numeric_perturbation_probe,realizable_star_probe
from .config import HISTORICAL,ROOT,named_seed
from .evaluation import load_spec,rollout
from .models import neighbor_reduce
from .report import csv
from .training import weights_hash

DIRECTORY=ROOT/"mechanisms"


def patterns(k,count,pattern):
    if pattern=="LOW":order=np.arange(k)
    elif pattern=="HIGH":order=np.arange(k-1,-1,-1)
    elif pattern=="ALTERNATING":order=np.r_[np.arange(0,k,2),np.arange(1,k,2)]
    elif pattern=="EXTREMES":
        order=np.array([j for i in range((k+1)//2) for j in (i,k-1-i)])
        order=np.array(list(dict.fromkeys(order.tolist())))
    elif pattern=="SCATTERED":order=np.random.default_rng(named_seed("patterns",k,count,pattern)).permutation(k)
    else:raise ValueError(pattern)
    return np.sort(order[:count])


@torch.inference_mode()
def center_state(model,colors,k):
    """Learned first communication in a star, retaining multiplicities."""
    device=next(model.parameters()).device
    dtype=next(model.parameters()).dtype
    colors=torch.tensor(colors,dtype=torch.long,device=device)
    n=len(colors)
    slots=torch.arange(k,device=device)
    own=(colors[:,None]==slots).to(dtype)
    u=(colors<0).to(dtype)
    f=torch.stack((u[:,None].expand(n,k),own),-1)
    messages=model.message1(f)
    reduction=getattr(model,"reduction","max")
    if reduction=="max":
        received=messages[1:].amax(0) if n>1 else torch.zeros_like(messages[0])
    else:
        # Use the actual original reducer on the star so SUM's summation order
        # agrees with the corresponding full-forward diagnostic.
        leaves=torch.arange(1,n,device=device)
        edges=torch.stack((leaves,torch.zeros_like(leaves)))
        received=neighbor_reduce(messages.reshape(n,-1),edges,reduction)[0].reshape(k,-1)
    h=model.state1(torch.cat((u[0].expand(k,1),received),-1))
    alpha=torch.sigmoid(model.availability(h).squeeze(-1))
    khat=alpha.sum()
    prefix=alpha.cumsum(0)-alpha
    lo,hi=prefix/khat.clamp_min(.5),(prefix+alpha)/khat.clamp_min(.5)
    return h,alpha,khat,prefix,lo,hi


@torch.inference_mode()
def choices(model,state,grid):
    h,alpha,khat,prefix,lo,hi=state
    k=len(alpha)
    # Full attention's temporary quadratic palette memory is bounded here.
    attention=getattr(model,"treatment","")=="transformer"
    chunk=max(1,min(128,8_000_000//max(1,2*k*k))) if attention else 128
    all_choices=[]
    for start in range(0,len(grid),chunk):
        r=torch.tensor(grid[start:start+chunk],dtype=h.dtype,device=h.device)
        b=len(r)
        broadcast=lambda x:x[None,:].expand(b,k)
        if hasattr(model,"palette_scores"):
            scores=model.palette_scores(h[None,:,:].expand(b,k,-1),broadcast(alpha),r,
                  torch.full((b,),k,device=h.device,dtype=torch.long),torch.ones((b,k),device=h.device,dtype=torch.bool))
        else:
            extra=(broadcast(alpha),broadcast(lo),broadcast(hi),r[:,None].expand(b,k)) if model.variant=="NORMALIZED" else (
                  broadcast(alpha),broadcast(prefix),broadcast(prefix+alpha),khat.expand(b,k),r[:,None].expand(b,k))
            scores=model.scorer(torch.cat((h[None,:,:].expand(b,k,-1),torch.stack(extra,-1)),-1)).squeeze(-1)
        all_choices.append(scores.argmax(1).cpu().numpy())
    return np.concatenate(all_choices)


def grids(count):
    bulk=((np.arange(2049)+.5)/2049).astype(np.float32)
    boundaries=(np.arange(1,count,dtype=np.float64)/count).astype(np.float32)
    near=np.unique(np.r_[np.nextafter(boundaries,np.float32(0)),boundaries,np.nextafter(boundaries,np.float32(1)),
                           np.float32(0),np.nextafter(np.float32(1),np.float32(0))]).astype(np.float32)
    return bulk,near


def candidate_metrics(selected,available,k,rc,predicted_rank=None):
    q=np.bincount(selected,minlength=k)/len(selected)
    target=np.zeros(k)
    target[available]=1/len(available)
    allowed=np.isin(selected,available)
    # Insertion rank is not treated as a valid rank for forbidden candidates.
    rank=np.searchsorted(available,selected).clip(0,len(available)-1)
    teacher_rank=np.floor(rc.astype(np.float64)*len(available)).astype(int).clip(0,len(available)-1)
    positive=q[q>0]
    row={"draws":len(selected),"teacher_fidelity":float((selected==available[teacher_rank]).mean()),
         "forbidden_rate":float((~allowed).mean()),"tv_uniform":float(.5*np.abs(q-target).sum()),
         "entropy":float(-(positive*np.log(positive)).sum()),"effective_colors":float(np.exp(-(positive*np.log(positive)).sum())),
         "max_mass":float(q.max()),"pair_collision":float((q*q).sum()),"distinct_colors":int((q>0).sum()),
         "candidate_monotone":bool(np.all(np.diff(selected)>=0)),"allowed_draws":int(allowed.sum()),
         "mean_teacher_rank_error_conditional_allowed":float(np.abs(rank[allowed]-teacher_rank[allowed]).mean()) if allowed.any() else None,
         "max_teacher_rank_error_conditional_allowed":int(np.abs(rank[allowed]-teacher_rank[allowed]).max()) if allowed.any() else None}
    if predicted_rank is not None:
        row.update(prediction_fidelity=float((selected==available[predicted_rank]).mean()),
                   mean_prediction_rank_error_conditional_allowed=float(np.abs(rank[allowed]-predicted_rank[allowed]).mean()) if allowed.any() else None,
                   max_prediction_rank_error_conditional_allowed=int(np.abs(rank[allowed]-predicted_rank[allowed]).max()) if allowed.any() else None)
    return row,q


def freeze_calibration(model,checkpoint):
    path=DIRECTORY/"frozen-quantile-calibration.json"
    checkpoint_hash=file_hash(checkpoint)
    if path.exists():
        result=json.loads(path.read_text())
        if result["checkpoint_sha256"]!=checkpoint_hash:raise RuntimeError("Calibration checkpoint changed")
        return result
    rc=((np.arange(257)+.5)/257).astype(np.float32)
    selected=choices(model,center_state(model,np.full(101,-1),101),rc)
    raw=selected/101.
    quantiles=np.maximum.accumulate(raw)
    result={"checkpoint":str(checkpoint),"checkpoint_sha256":checkpoint_hash,"calibration_palette":101,
            "pattern":"all free, star degree 100","r":rc.tolist(),"selected":selected.tolist(),
            "g":quantiles.tolist(),"raw_monotone":bool(np.all(np.diff(selected)>=0)),
            "definition":"monotone accumulation of chosen zero-based rank / 101, linear interpolation, endpoint clipping",
            "heldout_not_evaluated_yet":True,"frozen_utc":now(),"device":str(next(model.parameters()).device),
            "dtype":str(next(model.parameters()).dtype)}
    atomic_json(path,result)
    # This durable write precedes every held-out choice or score.
    return result


def selection_probe(model,spec,heldout=False,calibration=None,stop_requested=lambda:False,analytic=None):
    label=spec["label"]
    output=DIRECTORY/"selection"/label
    output.mkdir(parents=True,exist_ok=True)
    palettes=(17,63,251,997) if heldout else (4,11,21,51,101,201)
    for k in palettes:
        counts=sorted({v for v in (1,2,4,math.ceil(k/4),math.ceil(k/2),k) if v<=k})
        for count in counts:
            for pattern in ("LOW","HIGH","SCATTERED","ALTERNATING","EXTREMES"):
                if stop_requested():return
                stem=f"K{k}-k{count}-{pattern}"
                path=output/f"{stem}.json"
                if path.exists():continue
                available=patterns(k,count,pattern)
                occupied=np.setdiff1d(np.arange(k),available)
                colors=np.full(k,-1,np.int64)
                colors[1:1+len(occupied)]=occupied
                bulk,boundary=grids(count)
                predicted=None
                secondary={}
                if calibration is not None:
                    g=np.interp(bulk,calibration["r"],calibration["g"])
                    predicted=np.floor(count*g).astype(int).clip(0,count-1)
                    # Save predictions before reading actual held-out choices.
                    if analytic is not None:
                        from .predictors import predict_ranks
                        secondary=predict_ranks(analytic,bulk,count,colors)
                    np.savez_compressed(output/f"{stem}-prediction.npz",rc=bulk,available=available,predicted_rank=predicted,**secondary)
                state=center_state(model,colors,k)
                selected=choices(model,state,bulk)
                boundary_choices=choices(model,state,boundary)
                metrics,q=candidate_metrics(selected,available,k,bulk,predicted)
                for name,ranks in secondary.items():
                    comparison,_=candidate_metrics(selected,available,k,bulk,ranks)
                    metrics.update({name+"_"+key:value for key,value in comparison.items() if "prediction" in key})
                boundary_truth=available[np.floor(boundary.astype(np.float64)*count).astype(int).clip(0,count-1)]
                frontiers=(np.arange(1,count,dtype=np.float64)/count).astype(np.float32)
                below=np.searchsorted(boundary,np.nextafter(frontiers,np.float32(0)))
                above=np.searchsorted(boundary,np.nextafter(frontiers,np.float32(1)))
                alpha=state[1].cpu().numpy()
                relative=alpha/max(float(alpha.max()),1e-20)
                exact=np.isin(np.arange(k),available)
                row={"label":label,"treatment":spec["treatment"],"seed":spec["seed"],"palette":k,"available_count":count,"pattern":pattern,
                     "heldout_prediction":heldout,**metrics,"boundary_draws":len(boundary),
                     "boundary_fidelity":float((boundary_choices==boundary_truth).mean()),
                     "boundary_switch_rate":float((boundary_choices[below]!=boundary_choices[above]).mean()) if len(frontiers) else None,
                     "boundary_forbidden_rate":float((~np.isin(boundary_choices,available)).mean()),
                     "availability_slot_errors":int(((relative>=.5)!=exact).sum()),
                     "raw_availability_mass":float(state[2]),"calibration_checkpoint_sha256":calibration["checkpoint_sha256"] if calibration else None}
                row.update(device=str(next(model.parameters()).device),dtype=str(next(model.parameters()).dtype))
                atomic_json(path,row)
                np.savez_compressed(output/f"{stem}.npz",rc=bulk,choices=selected,boundary_rc=boundary,boundary_choices=boundary_choices,
                                    available=available,colors=colors,q=q,alpha=alpha)
        print(f"selection {label} K={k} heldout={heldout}",flush=True)


@torch.inference_mode()
def multiplicity_probe(model,spec):
    types=torch.tensor(TYPES,dtype=next(model.parameters()).dtype,device=next(model.parameters()).device)
    messages=model.message2(types)
    center=torch.tensor([1,0,1,1],dtype=types.dtype,device=types.device)
    rows=[]
    for conflict in (None,1,5):
        for ordinary_count in (1,2,8,64,1024):
            counts=torch.zeros(6,dtype=types.dtype,device=types.device)
            counts[2]=ordinary_count
            if conflict is not None:counts[conflict]=1
            reduction=getattr(model,"reduction","max")
            if reduction=="max":r=messages[counts>0].amax(0)
            else:
                r=(messages*counts[:,None]).sum(0)
                if reduction=="mean":r=r/counts.sum()
            h=model.state2(torch.cat((center,r)))
            logit=float(model.commit_head(torch.cat((h,torch.ones(3,dtype=types.dtype,device=types.device)))).squeeze())
            valid=conflict is None
            rows.append({"label":spec["label"],"reduction":reduction,"conflict_type":conflict,"ordinary_type_count":ordinary_count,
                         "degree":int(counts.sum()),"palette_for_realizable_star":int(counts.sum())+1,
                         "logit":logit,"valid":valid,"correct":bool((logit>0)==valid),"unsafe_accept":bool(logit>0 and not valid)})
    return rows


def progress_states():
    for core_degree in (3,10,20,50):
        for k in (2,4,11,21,51):
            if k<core_degree+1:continue
            for fixed in (0,8):
                m=core_degree+1
                palette=k+fixed
                n=m*(fixed+1)
                colors=np.full(n,-1,np.int64)
                src,dst=[],[]
                for u in range(m):
                    for v in range(u+1,m):src.extend((u,v));dst.extend((v,u))
                    for j in range(fixed):
                        leaf=m+u*fixed+j
                        colors[leaf]=k+j
                        src.extend((u,leaf));dst.extend((leaf,u))
                edges=np.array([src,dst],np.int64)
                meta={"state_id":f"core{core_degree}-free{k}-fixed{fixed}","core_degree":core_degree,"pending_core_nodes":m,
                      "available_count":k,"fixed_neighbors":fixed,"palette":palette,"n":n,
                      "actual_max_degree":core_degree+fixed,"declared_degree_bound":palette-1}
                yield meta,colors,edges


def progress_probe(model,spec,stop_requested=lambda:False):
    output=DIRECTORY/"progress"/spec["label"]
    output.mkdir(parents=True,exist_ok=True)
    for meta,colors,edges in progress_states():
        k,palette=meta["available_count"],meta["palette"]
        state_file=output/f"{meta['state_id']}-law.json"
        if not state_file.exists():
            # All pending clique vertices have this same local state.
            local=np.r_[-1,np.full(meta["core_degree"],-1),np.arange(k,palette)]
            grid,_=grids(k)
            selected=choices(model,center_state(model,local,palette),grid)
            _,q=candidate_metrics(selected,np.arange(k),palette,grid)
            with torch.inference_mode():
                rp=torch.tensor(grid,device=next(model.parameters()).device)
                prob=float((model.participation(torch.stack((rp,torch.ones_like(rp)),-1)).squeeze(-1)>0).float().mean())
            predicted=prob*np.sum(q[:k]*(1-prob*q[:k])**meta["core_degree"])
            atomic_json(state_file,{**meta,"label":spec["label"],"p_grid":prob,"q_grid":q.tolist(),
                       "predicted_safe_proposals_first_round":float(meta["pending_core_nodes"]*predicted),
                       "prediction_assumptions":"private independent node draws, p independent of C, exact conflict rejection; q,p approximated by 2049 midpoint grid"})
        else:predicted=json.loads(state_file.read_text())["predicted_safe_proposals_first_round"]/meta["pending_core_nodes"]
        for tape in range(16):
            if stop_requested():return
            path=output/f"{meta['state_id']}-t{tape:02d}.json"
            if path.exists():continue
            seed=named_seed("progress",meta["state_id"],tape)
            result,final=rollout(model,meta["n"],edges,palette,seed,initial_colors=colors,
                                 cyclic=spec["treatment"]=="cyclic_teacher")
            row={**meta,**result,"label":spec["label"],"treatment":spec["treatment"],"tape":tape,"tape_seed":seed,
                 "predicted_safe_proposals_first_round":float(meta["pending_core_nodes"]*predicted)}
            atomic_json(path,row)
        print(f"progress {spec['label']} {meta['state_id']}",flush=True)


def dense_flops(model,n,k,directed_edges):
    def linear_macs(module):
        return sum(layer.in_features*layer.out_features for layer in module if isinstance(layer,torch.nn.Linear))
    per_slot=sum(linear_macs(getattr(model,name)) for name in ("message1","state1","message2","state2"))+32
    kind=getattr(model,"treatment","normalized")
    attention=0
    if kind=="deepsets":per_slot+=linear_macs(model.palette_phi)+linear_macs(model.scorer)
    elif kind=="transformer":
        per_slot+=36*16+4*16*16+2*16*32+linear_macs(model.scorer)
        attention=4*n*k*k*16
    else:per_slot+=linear_macs(model.scorer)
    per_node=linear_macs(model.participation)+linear_macs(model.commit_head)
    return {"approx_dense_flops":2*(n*k*per_slot+n*per_node)+attention,
            "attention_quadratic_flops":attention,"neighbor_comparisons_or_additions":2*directed_edges*k*32,
            "logical_message_bytes_per_round":2*directed_edges*k*32*4,
            "exclusions":"nonlinearities, softmax, normalization, padding and gather overhead; multiply/add count separately as two FLOPs"}


@torch.inference_mode()
def resource_probe(model,spec):
    from dcl.extreme.graphs import generate
    from dcl.m4_data import _edge_index
    device=next(model.parameters()).device
    rows=[]
    for d in (3,10,20,50,100,200):
        k,n=d+1,max(160,2*d)
        graph=generate(n,d,"REGULAR",named_seed("resource",d,n))
        edges=torch.tensor(_edge_index(graph),device=device)
        colors=torch.full((n,),-1,dtype=torch.long,device=device)
        rp=torch.full((n,),.25,device=device)
        rc=torch.full((n,),.63,device=device)
        palette=torch.full((n,),k,dtype=torch.long,device=device)
        try:
            model(colors,rp,rc,edges,palette,8192)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick=time.perf_counter()
            for _ in range(5):model(colors,rp,rc,edges,palette,8192)
            torch.cuda.synchronize()
            row={"status":"ok","forward_seconds":(time.perf_counter()-tick)/5,"peak_gpu_bytes":int(torch.cuda.max_memory_allocated())}
        except torch.OutOfMemoryError as error:
            row={"status":"oom","error":str(error)}
            torch.cuda.empty_cache()
        rows.append({"label":spec["label"],"treatment":spec["treatment"],"n":n,"degree":d,"palette":k,"directed_edges":edges.shape[1],
                     "parameters":sum(p.numel() for p in model.parameters()),**dense_flops(model,n,k,edges.shape[1]),**row})
    return rows


def export_mechanisms():
    output=Path("results/ucgnn-paper-study")
    selection=[]
    for p in (DIRECTORY/"selection").glob("*/*.json"):selection.append(json.loads(p.read_text()))
    csv(output/"candidate-diagnostics.csv",selection)
    progress,rounds=[],[]
    for p in (DIRECTORY/"progress").glob("*/*.json"):
        if p.stem.endswith("-law"):continue
        r=json.loads(p.read_text())
        for t in r.pop("round_trace",[]):rounds.append({"label":r["label"],"state_id":r["state_id"],"tape":r["tape"],**t})
        r.pop("by_k",None)
        progress.append(r)
    csv(output/"progress-diagnostics.csv",progress)
    csv(output/"progress-round-traces.csv",rounds)
    csv(output/"progress-counterexample.csv",[{"clique_nodes":k,"per_node_confirmation_probability":2.**-k,
           "probability_any_confirmation":k*2.**-k,"expected_rounds_to_first_confirmation":2.**k/k} for k in (4,11,21,51,101,201)])


def schedule_tasks():
    paths=[]
    for p in sorted((ROOT/"models").glob("*.json")):
        s=json.loads(p.read_text())
        if s["selection"]=="epoch120" and s["kind"]!="specialist":paths.append((p,s))
    paths.sort(key=lambda pair:(pair[1]["label"]!="historical-e1-s0-epoch120",pair[1]["label"]))
    tasks=[{"id":f"mechanisms-{s['label']}","action":"mechanisms","status":"pending","attempts":[],"timeout":1800,
            "model_spec":str(p),"arguments":["mechanisms","--model-spec",str(p)]} for p,s in paths]
    tasks.append({"id":"mechanisms-finalize","action":"mechanisms","status":"pending","attempts":[],"timeout":1800,"arguments":["mechanisms"]})
    return tasks


def run_all(stop_requested=lambda:False,only_spec=None):
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    specs=[json.loads(p.read_text()) for p in sorted((ROOT/"models").glob("*.json"))]
    epoch=[s for s in specs if s["selection"]=="epoch120" and s["kind"]!="specialist"]
    if only_spec is not None:
        epoch=[json.loads(Path(only_spec).read_text())]
    certificates,multiplicity,resources=[],[],[]
    for spec in epoch:
        if stop_requested():return
        complete_path=DIRECTORY/"completed"/f"{spec['label']}.json"
        if complete_path.exists():continue
        model,_=load_spec(spec)
        whash=weights_hash(model.state_dict())
        if spec["label"]=="historical-e1-s0-epoch120":
            from .predictors import freeze_analytic
            calibration=freeze_calibration(model,spec["checkpoint"])
            analytic=freeze_analytic(model,spec["checkpoint"],DIRECTORY/"frozen-analytic-quantiles.json")
            selection_probe(model,spec,True,calibration,stop_requested,analytic)
        real=commit_certificate(model)
        numerical=commit_certificate(model,True)
        certificate={"label":spec["label"],"checkpoint_sha256":spec["checkpoint_sha256"],"real":real,"float32_model":numerical}
        certificate["actual_realizable_stars"]=realizable_star_probe(model)
        if numerical.get("applicable"):
            for r in certificate["actual_realizable_stars"]:
                interval=numerical["rows"][r["mask"]]
                r["inside_float32_error_enclosure"]=interval["logit_lower"]<=r["actual_logit"]<=interval["logit_upper"]
        if real["applicable"]:
            certificate["numeric_float32"]=numeric_perturbation_probe(model)
            model.double()
            certificate["numeric_float64"]=numeric_perturbation_probe(model)
            model.float()
        atomic_json(DIRECTORY/"certificates"/f"{spec['label']}.json",certificate)
        certificates.append({"label":spec["label"],"treatment":spec["treatment"],"seed":spec["seed"],"finite_max_applicable":real["applicable"],
                             "real_exact_rule_certified":real.get("certified_exact_rule"),"float32_model_exact_rule_certified":numerical.get("certified_exact_rule"),
                             "real_margin_lower":real.get("min_signed_margin_lower"),"unsafe_masks_certified":len(real.get("certified_unsafe_masks",[]))})
        multiplicity.extend(multiplicity_probe(model,spec))
        # All new seeds get all-free laws; seed 41000 and historical controls
        # get the full fixed pattern battery. No performance-based selection.
        full=spec["seed"] in (0,41000,41005)
        if full:
            selection_probe(model,spec,stop_requested=stop_requested)
            resources.extend(resource_probe(model,spec))
            progress_probe(model,spec,stop_requested)
        else:
            for k in (4,11,21,51,101,201):
                grid,_=grids(k)
                selected=choices(model,center_state(model,np.full(k,-1),k),grid)
                metrics,_=candidate_metrics(selected,np.arange(k),k,grid)
                atomic_json(DIRECTORY/"selection"/spec["label"]/f"K{k}-k{k}-LOW.json",
                            {"label":spec["label"],"treatment":spec["treatment"],"seed":spec["seed"],"palette":k,
                             "available_count":k,"pattern":"LOW","heldout_prediction":False,**metrics})
        if weights_hash(model.state_dict())!=whash:raise RuntimeError("Diagnostics changed checkpoint weights")
        del model
        torch.cuda.empty_cache()
        if stop_requested():return
        atomic_json(complete_path,{"complete":True,"label":spec["label"],"utc":now(),
                    "certificate_summary":certificates[-1],
                    "multiplicities":[r for r in multiplicity if r["label"]==spec["label"]],
                    "resources":[r for r in resources if r["label"]==spec["label"]]})
        export_mechanisms()
    completed=[json.loads(p.read_text()) for p in (DIRECTORY/"completed").glob("*.json")]
    csv("results/ucgnn-paper-study/commit-certificates.csv",[r["certificate_summary"] for r in completed])
    csv("results/ucgnn-paper-study/commit-multiplicities.csv",[r for v in completed for r in v["multiplicities"]])
    csv("results/ucgnn-paper-study/resources.csv",[r for v in completed for r in v["resources"]])
    if only_spec is not None:return
    atomic_json(DIRECTORY/"complete.json",{"complete":True,"utc":now(),"models":len(completed),
                "limits":"Patterns at held-out K=17,63,251,997 are the frozen E1-0 prediction experiment; new-model patterned battery capped at K=201. Grid probabilities are approximate."})
