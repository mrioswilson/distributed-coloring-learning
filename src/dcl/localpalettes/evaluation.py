from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from dcl.study.storage import atomic_json
from dcl.universal.rules import available_matrix, conflict_flags, exact_commit_labels, teacher_offer
from .adapter import LocalPaletteAdapter, converted_uniform, first_state, choices
from .cohort import palette_for, star_edges
from .config import ROOT, seed
from .statistics import law_metrics, likelihood_curve, product_bounds, uncertain_product_bounds, wilson


def draw_tape(graph_id,tape,n,rounds=256):
    s=seed('tape',graph_id,tape)
    # One frozen layout [round,node,(rp,rc)], same for all models/regimes.
    return np.random.default_rng(s).random((rounds,n,2)),s


def rollout(model,edges,degrees,regime,graph_id,tape,rounds=256,initial=None,capture_round=None):
    palette=palette_for(regime,degrees)
    n=len(degrees)
    colors=np.full(n,-1,dtype=np.int64) if initial is None else np.array(initial).copy()
    random,s=draw_tape(graph_id,tape,n,rounds)
    adapter=LocalPaletteAdapter(model) if model is not None else None
    device=next(model.parameters()).device if model is not None else 'cpu'
    e=torch.as_tensor(edges,device=device)
    pal=torch.as_tensor(palette,device=device)
    trace=[];totals=dict(unsafe_commits=0,out_of_list=0,forbidden_candidates=0,proposals=0,participations=0,simultaneous=0)
    captured=None
    start=time.monotonic()
    for r in range(rounds):
        pending=colors<0
        if not pending.any():break
        rp,rc=random[r,:,0],random[r,:,1]
        if model is None:
            cand,_=teacher_offer(colors,rc,edges,palette)
            part=(rp<.5)&pending
            commit=exact_commit_labels(colors,edges,part,cand)
        else:
            output=adapter(torch.as_tensor(colors,device=device),torch.as_tensor(converted_uniform(rp),device=device),
                           torch.as_tensor(converted_uniform(rc),device=device),e,pal)
            cand=output.candidate.cpu().numpy();part=output.participate.cpu().numpy()
            commit=(output.commit_logit>0).cpu().numpy()&part&pending
        outside=pending&((cand<0)|(cand>=palette))
        if outside.any():
            # An implementation failure is retained; never repair/mask an action.
            return dict(status='implementation_failure',rounds=r,out_of_list=int(outside.sum()),trace=trace,tape_seed=s),colors,captured
        exact=exact_commit_labels(colors,edges,part,cand)
        fixed,simultaneous=conflict_flags(colors,edges,part,cand)
        chosen=cand[pending]
        _,counts=np.unique(chosen,return_counts=True)
        proposals=int(pending.sum())
        collision_pairs=int(np.sum(counts*(counts-1)//2))
        row=dict(round=r+1,pending=proposals,participations=int(part.sum()),commits=int(commit.sum()),
                 unsafe_commits=int((commit&~exact).sum()),forbidden_candidates=int((pending&fixed).sum()),
                 safe_proposals=int((pending&~fixed).sum()),simultaneous=int((pending&part&simultaneous).sum()),
                 distinct=int(len(counts)),max_candidate_mass=float(counts.max()/proposals),
                 pair_collisions=collision_pairs)
        trace.append(row)
        for key in totals:
            totals[key]+=proposals if key=='proposals' else row.get(key,0)
        colors[commit]=cand[commit]
        if capture_round==r+1:captured=colors.copy()
    src,dst=edges
    conflicts=int(np.sum((colors[src]>=0)&(colors[src]==colors[dst]))//2)
    remaining=int(np.sum(colors<0))
    status='success' if remaining==0 and conflicts==0 else ('unsafe_failure' if conflicts or totals['unsafe_commits'] else 'progress_cutoff')
    result=dict(status=status,rounds=len(trace),pending=remaining,conflicts=conflicts,
                **totals,tape_seed=s,trace=trace,seconds=time.monotonic()-start,
                candidate_histogram_kind='round summaries; controlled-state full histograms separately')
    return result,colors,captured


def center_cache(model,state):
    device=next(model.parameters()).device
    colors=torch.tensor(state['colors'],device=device)
    edges=torch.tensor(state.get('edges',star_edges(len(colors))),device=device)
    center=int(state.get('center',0))
    h,a=first_state(model,colors,edges,torch.tensor([center],device=device),int(state['k']))
    return h[0],a[0]


def cyclic_law(available,k):
    a=np.array(available,dtype=np.int64)
    gaps=(a-np.roll(a,1))%k
    if len(a)==1:gaps[:]=k
    out=np.zeros(k);out[a]=gaps/k
    return out


def evaluate_law(model,model_label,state,directory):
    """All splits have distinct named RNG streams, retained with sample counts."""
    k=int(state['k']);available=np.array(state['available'])
    state_id=state['id'];cache=center_cache(model,state)
    seeds={name:seed('candidate',model_label,state_id,name) for name in ('estimation','test','b3-model','b3-teacher','calibration','ties')}
    est_rc=np.random.default_rng(seeds['estimation']).random(32768)
    test_rc=np.random.default_rng(seeds['test']).random(8192)
    est=choices(model,cache,converted_uniform(est_rc))
    test=choices(model,cache,converted_uniform(test_rc))
    counts=np.bincount(est,minlength=k)
    test_counts=np.bincount(test,minlength=k)
    teacher=available[np.floor(test_rc*len(available)).astype(int)]
    teacher32=available[np.floor(converted_uniform(test_rc).astype(np.float64)*len(available)).astype(int)]
    grid=(np.arange(2049)+.5)/2049
    grid_choices=choices(model,cache,converted_uniform(grid))
    boundaries=(np.arange(1,len(available),dtype=np.float64)/len(available)).astype(np.float32)
    near=np.unique(np.r_[np.float32(0),np.nextafter(np.float32(1),np.float32(0)),boundaries,
                         np.nextafter(boundaries,np.float32(0)),np.nextafter(boundaries,np.float32(1))])
    near=near[(near>=0)&(near<1)]
    near_choices=choices(model,cache,near)
    metric=law_metrics(counts,available)
    fidelity=int(np.sum(test==teacher32))
    allowed=np.isin(test,available)
    rank=np.searchsorted(available,test[allowed])
    teacher_rank=np.floor(converted_uniform(test_rc[allowed]).astype(float)*len(available)).astype(int)
    cyc=cyclic_law(available,k)
    q=counts/counts.sum()
    metric.update(model=model_label,state=state_id,k=k,pattern=state['pattern'],kind=state['kind'],
                  fidelity=fidelity/len(test),fidelity_interval=wilson(fidelity,len(test)),
                  fidelity_float64_teacher=float(np.mean(test==teacher)),
                  teacher_float_precision_disagreement=float(np.mean(teacher!=teacher32)),
                  rank_error=float(np.mean(np.abs(rank-teacher_rank))) if len(rank) else None,
                  cyclic_tv=float(.5*np.abs(q-cyc).sum()),estimation_n=len(est),test_n=len(test),
                  quadrature_tv=law_metrics(np.bincount(grid_choices,minlength=k),available)['tv'],
                  seeds=seeds,rng_algorithm='PCG64',numpy_version=np.__version__,
                  rng_initial_states={name:np.random.default_rng(s).bit_generator.state for name,s in seeds.items()})
    b3=[]
    if state.get('b3'):
        rng=np.random.default_rng(seeds['b3-model'])
        rc=rng.random((256,2000))
        model_test=choices(model,cache,converted_uniform(rc.reshape(-1))).reshape(256,2000)
        teacher_test=np.random.default_rng(seeds['b3-teacher']).choice(available,size=(256,2000))
        b3=likelihood_curve(counts,available,model_test,teacher_test,np.random.default_rng(seeds['calibration']),tie_rng=np.random.default_rng(seeds['ties']))
        for row in b3:
            row.update(model=model_label,state=state_id,k=k,pattern=state['pattern'],
                       estimated_law_product_bounds=product_bounds(metric['tv'],metric['affinity'],row['m']),
                       true_law_product_uncertainty=uncertain_product_bounds(metric,row['m']))
        np.savez_compressed(directory/f'{state_id}-b3.npz',model=model_test.astype(np.uint16),teacher=teacher_test.astype(np.uint16))
    np.savez_compressed(directory/f'{state_id}.npz',estimation_counts=counts,test_counts=test_counts,
                        quadrature_rc=grid,quadrature_choices=grid_choices,boundary_rc=near,boundary_choices=near_choices,
                        available=available,cyclic_law=cyc)
    return dict(metrics=metric,b3=b3)
