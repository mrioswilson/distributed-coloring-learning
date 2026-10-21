"""Outward interval certificates for the finite MAX COMMIT channel.

Real-coefficient mode encloses every elementary operation in float64. The
float32-error mode additionally assumes each affine evaluation satisfies the
standard gamma_(n+2) dot-product bound, IEEE rounding, and no overflow/TF32.
These assumptions are part of the certificate, not observations about a GPU.
"""
from __future__ import annotations

import numpy as np
import torch

TYPES = np.array([[0,0,0,0],[0,1,0,0],[1,0,0,0],
                  [1,0,0,1],[1,0,1,0],[1,0,1,1]],dtype=np.float64)


def down(x):
    return np.nextafter(np.asarray(x,dtype=np.float64),-np.inf)


def up(x):
    return np.nextafter(np.asarray(x,dtype=np.float64),np.inf)


def affine_interval(lo,hi,weight,bias=None,float32_error=False):
    lo,hi,weight = np.asarray(lo,np.float64),np.asarray(hi,np.float64),np.asarray(weight,np.float64)
    shape = (*lo.shape[:-1],weight.shape[0])
    initial = np.zeros(weight.shape[0]) if bias is None else np.asarray(bias,np.float64)
    lower,upper = np.broadcast_to(initial,shape).copy(),np.broadcast_to(initial,shape).copy()
    absolute_bound = up(np.broadcast_to(np.abs(initial),shape).copy())
    for j in range(weight.shape[1]):
        a,b = lo[...,j,None]*weight[:,j],hi[...,j,None]*weight[:,j]
        lower = down(lower + down(np.minimum(a,b)))
        upper = up(upper + up(np.maximum(a,b)))
        term = up(np.maximum(np.abs(lo[...,j,None]),np.abs(hi[...,j,None]))*np.abs(weight[:,j]))
        absolute_bound = up(absolute_bound+term)
    if float32_error:
        m = weight.shape[1]+2
        unit = 2.**-24
        gamma = up((m*unit)/(1-m*unit))
        error = up(up(gamma*absolute_bound) + up(m * 2.**-149))
        lower,upper = down(lower-error),up(upper+error)
        if np.any(up(absolute_bound+error) >= np.finfo(np.float32).max):
            raise OverflowError("Cannot certify float32 without excluding overflow")
    return lower,upper


def module_interval(module,lo,hi,float32_error=False):
    for layer in module:
        if isinstance(layer,torch.nn.Linear):
            weight = layer.weight.detach().cpu().double().numpy()
            bias = layer.bias.detach().cpu().double().numpy() if layer.bias is not None else None
            lo,hi = affine_interval(lo,hi,weight,bias,float32_error)
        elif isinstance(layer,torch.nn.ReLU):
            lo,hi = np.maximum(lo,0),np.maximum(hi,0)
        else:
            raise TypeError(f"Unsupported interval layer {layer}")
    return lo,hi


def commit_certificate(model,float32_error=False):
    if getattr(model,"reduction","max")!="max":
        return {"applicable":False,"reason":"64 sets do not cover arbitrary SUM/MEAN multiplicities"}
    mlo,mhi = module_interval(model.message2,TYPES,TYPES,float32_error)
    rows = []
    center = np.array([1.,0.,1.,1.])
    for mask in range(64):
        indices = [i for i in range(6) if mask&(1<<i)]
        rlo = np.maximum(0,mlo[indices].max(0)) if indices else np.zeros(mlo.shape[1])
        rhi = np.maximum(0,mhi[indices].max(0)) if indices else np.zeros(mhi.shape[1])
        zlo,zhi = module_interval(model.state2,np.r_[center,rlo],np.r_[center,rhi],float32_error)
        clo,chi = module_interval(model.commit_head,np.r_[zlo,1.,1.,1.],np.r_[zhi,1.,1.,1.],float32_error)
        valid = not (mask&(1<<1) or mask&(1<<5))
        lo,hi = float(clo[0]),float(chi[0])
        certified_correct = lo>0 if valid else hi<0
        certified_unsafe = lo>0 and not valid
        rows.append({"mask":mask,"valid":valid,"logit_lower":lo,"logit_upper":hi,
                     "certified_correct":bool(certified_correct),"certified_unsafe":bool(certified_unsafe),
                     "correct_signed_margin_lower":lo if valid else -hi})
    return {"applicable":True,"arithmetic":"float32_error_model" if float32_error else "real_weights",
            "certified_exact_rule":all(r["certified_correct"] for r in rows),
            "certified_no_false_accept":all(r["logit_upper"]<0 for r in rows if not r["valid"]),
            "certified_unsafe_masks":[r["mask"] for r in rows if r["certified_unsafe"]],
            "min_signed_margin_lower":min(r["correct_signed_margin_lower"] for r in rows),"rows":rows,
            "assumptions":["legal six binary neighbor types","nonnegative learned messages","componentwise MAX",
                           "exact one-slot gather","strict logit signs","IEEE float64 outward enclosures"]+
                          (["each float32 affine obeys gamma_(n+2) error bound","no TF32/overflow","IEEE rounding"] if float32_error else [])}


@torch.inference_mode()
def numeric_perturbation_probe(model,epsilon=1e-6):
    """Signed-logit margins and a conservative infinity-norm Lipschitz radius."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    types = torch.tensor(TYPES,dtype=dtype,device=device)
    messages = model.message2(types)
    center = torch.tensor([1,0,1,1],dtype=dtype,device=device)
    # Bound |delta logit| <= L ||delta pooled-message||_inf using induced norms.
    w1 = model.state2[0].weight[:,4:]
    w2 = model.state2[2].weight
    w3 = model.commit_head[0].weight[:,:messages.shape[1]]
    w4 = model.commit_head[2].weight
    lipschitz = 1.
    for w in (w1,w2,w3,w4):
        values=np.abs(w.detach().cpu().double().numpy())
        row_sum=np.zeros(values.shape[0])
        for j in range(values.shape[1]):
            row_sum=up(row_sum+values[:,j])
        lipschitz=float(up(lipschitz*float(row_sum.max())))
    real=commit_certificate(model)
    rows=[]
    for mask in range(64):
        ids=[j for j in range(6) if mask&(1<<j)]
        r=messages[ids].amax(0) if ids else torch.zeros_like(messages[0])
        valid=not (mask&(1<<1) or mask&(1<<5))
        signs=[]
        for perturbation in (0.,-epsilon,epsilon):
            z=model.state2(torch.cat((center,r+perturbation)))
            logit=float(model.commit_head(torch.cat((z,torch.ones(3,dtype=dtype,device=device)))).squeeze())
            signs.append(logit)
        signed=real["rows"][mask]["correct_signed_margin_lower"]
        rows.append({"mask":mask,"valid":valid,"logit":signs[0],"minus_epsilon_logit":signs[1],
                     "plus_epsilon_logit":signs[2],"correct":bool((signs[0]>0)==valid),
                     "perturbed_both_correct":all((s>0)==valid for s in signs),
                     "certified_real_input_radius_by_lipschitz":float(max(0.,down(max(0.,signed)/max(lipschitz,1e-300))))})
    return {"epsilon":epsilon,"infinity_lipschitz_upper":lipschitz,"rows":rows}


@torch.inference_mode()
def realizable_star_probe(model):
    """Execute actual legal star inputs, not only a reduced type-table MLP."""
    device=next(model.parameters()).device
    rows=[]
    reduction=getattr(model,"reduction","max")
    types=torch.tensor(TYPES,dtype=next(model.parameters()).dtype,device=device)
    messages=model.message2(types)
    for mask in range(64):
        ids=[j for j in range(6) if mask&(1<<j)]
        n=1+len(ids)
        colors=[-1]
        participate=[True]
        candidates=[0]
        for t in ids:
            colors.append(1 if t==0 else 0 if t==1 else -1)
            participate.append(t in (4,5))
            candidates.append(0 if t in (3,5) else 1 if t==4 else -1)
        leaves=torch.arange(1,n,device=device)
        edges=torch.stack((torch.cat((torch.zeros_like(leaves),leaves)),torch.cat((leaves,torch.zeros_like(leaves)))))
        color_t=torch.tensor(colors,device=device)
        part_t=torch.tensor(participate,device=device)
        cand_t=torch.tensor(candidates,device=device)
        palette=torch.full((n,),7,dtype=torch.long,device=device)
        actual=float(model.commit_logits(color_t,edges,palette,part_t,cand_t)[0])
        if not ids:r=torch.zeros_like(messages[0])
        elif reduction=="max":r=messages[ids].amax(0)
        elif reduction=="sum":r=messages[ids].sum(0)
        else:r=messages[ids].mean(0)
        own=torch.tensor([1,0,1,1],dtype=messages.dtype,device=device)
        h=model.state2(torch.cat((own,r)))
        reduced=float(model.commit_head(torch.cat((h,torch.ones(3,dtype=messages.dtype,device=device)))).squeeze())
        valid=not (mask&(1<<1) or mask&(1<<5))
        rows.append({"mask":mask,"degree":len(ids),"palette":7,"valid":valid,"actual_logit":actual,
                     "reduced_logit":reduced,"absolute_logit_difference":abs(actual-reduced),
                     "same_decision":bool((actual>0)==(reduced>0)),"correct":bool((actual>0)==valid),
                     "unsafe_actual_commit":bool(actual>0 and not valid)})
    return rows


def main():
    import argparse
    from pathlib import Path
    from dcl.study.storage import atomic_json,file_hash
    from .config import HISTORICAL,ROOT
    from .training import load_model as paper_load
    from dcl.universal.train import load_model as old_load
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/ucgnn-paper-study/certificates")
    args=parser.parse_args()
    paths=[]
    for group in ("e1","c1","a1"):
        paths.extend((HISTORICAL/"data/ucgnn/runs").glob(f"{group}-*/epoch-120.pt"))
    paths.extend((ROOT/"runs").glob("*/epoch-120.pt"))
    for p in sorted(paths):
        loader=paper_load if str(p).startswith(str(ROOT)) else old_load
        model,_=loader(p,"cpu")
        value={"checkpoint":str(p),"checkpoint_sha256":file_hash(p),
               "real":commit_certificate(model),"float32_model":commit_certificate(model,True)}
        atomic_json(Path(args.output)/f"{p.parent.name}.json",value)
        print(p.parent.name,value["real"].get("certified_exact_rule"),flush=True)


if __name__=="__main__":
    main()
