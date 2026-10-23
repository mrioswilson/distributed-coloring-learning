"""Exploratory integration of one-input, one-hidden-ReLU score surrogates.

Frozen float32 features and weights are promoted to float64, and the resulting
real-arithmetic piecewise-affine score is integrated. This is a numerical
reference for E1/DeepSets, NOT an exact GPU-float32 probability certificate.
No result from this module enters the confirmatory test or sample allocation.
"""
import numpy as np
import torch
from .common import *
from dcl.localpalettes.adapter import first_state,score
from dcl.paperstudy.training import load_model

def line_envelope(slopes,intercepts):
    order=np.lexsort((np.arange(len(slopes)),-intercepts,slopes))
    lines=[];starts=[]
    for i in order:
        if lines and slopes[i]==slopes[lines[-1]]:continue
        x=-np.inf
        while lines:
            j=lines[-1];x=(intercepts[j]-intercepts[i])/(slopes[i]-slopes[j])
            if x>starts[-1]:break
            lines.pop();starts.pop()
        if not lines:x=-np.inf
        lines.append(int(i));starts.append(float(x))
    return lines,starts

def masses(A,B,w,bias=0.):
    """Integrate argmax_c sum_j w_j ReLU(A_cj + B_cj r) + bias."""
    A=np.asarray(A,float);B=np.broadcast_to(np.asarray(B,float),A.shape);w=np.asarray(w,float)
    roots=np.divide(-A,B,out=np.full_like(A,np.inf),where=B!=0)
    knots=np.unique(np.r_[0.,roots[(roots>0)&(roots<1)],1.])
    q=np.zeros(len(A));top_margin=np.inf;envelope_segments=0
    for lo,hi in zip(knots[:-1],knots[1:]):
        mid=(lo+hi)/2;active=(A+B*mid)>0
        slopes=(active*B*w).sum(1);intercepts=(active*A*w).sum(1)+bias
        lines,starts=line_envelope(slopes,intercepts)
        for index,c in enumerate(lines):
            left=max(lo,starts[index]);right=min(hi,starts[index+1] if index+1<len(lines) else np.inf)
            if right<=left:continue
            q[c]+=right-left;envelope_segments+=1
            gap_s=slopes[c]-slopes[-1];gap_b=intercepts[c]-intercepts[-1]
            top_margin=min(top_margin,gap_s*left+gap_b,gap_s*right+gap_b)
    assert abs(q.sum()-1)<1e-10 and (q>=0).all()
    return dict(q_reference=q.tolist(),top_reference_mass=float(q[-1]),
                top_minimum_dominance_gap=float(top_margin),relu_intervals=len(knots)-1,envelope_segments=envelope_segments)

@torch.inference_mode()
def reference(model,g,state):
    assert model.treatment in ('normalized','deepsets')
    pending=np.flatnonzero(state<0);v=int(pending[0]);k=g['degrees'][v]+1
    e=torch.tensor(g['edges']);h,a=first_state(model,torch.tensor(state),e,torch.tensor([v]),k)
    captured=[]
    hook=model.scorer.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    try:
        score(model,h,a,torch.zeros(1));score(model,h,a,torch.ones(1))
    finally:hook.remove()
    z=captured[0][0].double().numpy();delta=(captured[1][0]-captured[0][0]).double().numpy()
    # Only the explicit private-uniform input changes. Attention is excluded.
    assert np.all(np.sum(delta!=0,axis=1)==1) and np.all(delta[delta!=0]==1)
    first=model.scorer[0];last=model.scorer[2]
    W=first.weight.detach().double().numpy();b=first.bias.detach().double().numpy()
    A=z@W.T+b;B=delta@W.T;w=last.weight.detach().double().numpy()[0]
    bias=0. if last.bias is None else float(last.bias[0])
    return masses(A,B,w,bias)

def main():
    torch.set_num_threads(1);rows=[]
    initial=read(OUT/'cpu-initial-probes.json')
    for s in specs():
        if s['treatment']=='transformer':continue
        model,_=load_model(s['checkpoint'],'cpu')
        for d in DS:
            g=next(g for g in graphs() if g['stage']=='main' and g['n']==128 and g['delta']==d and g['rep']==0)
            q=reference(model,g,np.full(g['n'],-1,np.int64))
            grid=next(r for r in initial if r['model']==s['label'] and r['delta']==d)
            rows.append(dict(model=s['label'],delta=d,grid_top=grid['q_top'],
                             grid_reference_l1=float(np.abs(np.array(q['q_reference'])-grid['q']).sum()),**q))
        print(s['label'],[(r['delta'],r['top_reference_mass']) for r in rows if r['model']==s['label']],flush=True)
    atomic_json(OUT/'initial-scorer-reference-integration.json',dict(rows=rows,
                estimand='Real-arithmetic surrogate with frozen float32 features/weights promoted to float64; not the complete numerical GPU policy law',
                scope='exploratory candidate diagnostic; never used in confirmation'))

if __name__=='__main__':main()
