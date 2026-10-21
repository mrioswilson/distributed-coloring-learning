from __future__ import annotations

import numpy as np
import torch

from dcl.study.storage import atomic_json
from dcl.universal.rules import available_matrix, exact_commit_labels
from .adapter import LocalPaletteAdapter, converted_uniform, first_state, choices
from .config import ROOT, RESULTS, seed


def expected_confirmations(q,p,colors,edges,available):
    out=np.zeros(len(colors))
    src,dst=edges
    for v in np.flatnonzero(colors<0):
        neighbors=src[(dst==v)&(colors[src]<0)]
        survive=np.prod(1-p[neighbors,None]*q[neighbors],axis=0)
        out[v]=p[v]*np.sum(q[v]*survive*available[v])
    return out


def run_progress():
    from .runner import frozen_autonomous_states, specs, load
    clique=[];heterogeneous=[]
    for spec in specs():
        for k in (4,11,51,201):
            path=ROOT/'distributions'/spec['label']/f'k{k}-ALL_FREE.npz'
            if not path.exists():continue
            data=np.load(path);q=data['estimation_counts'].astype(float);q/=q.sum()
            for name,law in (('learned-estimated',q),('uniform',np.full(k,1/k)),('half-uniform-mixture',.5*q+.5/k)):
                probability=float(.5*np.sum(law*(1-.5*law)**(k-1)))
                clique.append(dict(model=spec['label'],k=k,intervention=name,p=.5,
                                   exact_commit_probability_plugin=probability,expected_commits=k*probability,
                                   collision=float(np.sum(law**2)),effective_colors=float(1/np.sum(law**2)),
                                   tv=float(.5*np.abs(law-1/k).sum()),
                                   interpretation='Formula exact for supplied categorical law; learned law is a finite-MC estimate'))
    for spec in [s for s in specs() if s['seed']==41000 and s['treatment'] in ('normalized','deepsets','transformer')]:
        model=load(spec);adapter=LocalPaletteAdapter(model)
        for state in frozen_autonomous_states():
            colors=np.array(state['colors']);edges=np.array(state['edges']);palette=np.array(state['palettes'])
            n=len(colors);width=int(max(palette));available=available_matrix(colors,edges,palette)
            q=np.zeros((n,width));p=np.zeros(n)
            ct=torch.tensor(colors,device='cuda');et=torch.tensor(edges,device='cuda');pt=torch.tensor(palette,device='cuda')
            for v in np.flatnonzero(colors<0):
                with torch.inference_mode():
                    h,a=first_state(model,ct,et,torch.tensor([v],device='cuda'),int(palette[v]))
                    rng=np.random.default_rng(seed('progress-marginal',spec['label'],state['id'],v))
                    candidate=choices(model,(h[0],a[0]),converted_uniform(rng.random(2048)))
                    q[v,:palette[v]]=np.bincount(candidate,minlength=int(palette[v]))/len(candidate)
                    rp=torch.tensor(converted_uniform(rng.random(8192)),device='cuda')
                    logit=model.participation(torch.stack((rp,torch.ones_like(rp)),-1)).squeeze(-1)
                    p[v]=float((logit>0).float().mean())
            formula=expected_confirmations(q,p,colors,edges,available)
            empirical_exact=[];empirical_learned=[];unsafe=[]
            rng=np.random.default_rng(seed('progress-test',spec['label'],state['id']))
            for _ in range(256):
                draws=converted_uniform(rng.random((n,2)))
                output=adapter(ct,torch.tensor(draws[:,0],device='cuda'),torch.tensor(draws[:,1],device='cuda'),et,pt)
                part=output.participate.cpu().numpy();cand=output.candidate.cpu().numpy()
                exact=exact_commit_labels(colors,edges,part,cand)
                learned=(output.commit_logit>0).cpu().numpy()&part&(colors<0)
                empirical_exact.append(int(exact.sum()));empirical_learned.append(int(learned.sum()));unsafe.append(int(np.sum(learned&~exact)))
            row=dict(model=spec['label'],state=state['id'],pending=int(np.sum(colors<0)),
                     expected_exact_commits_plugin=float(formula.sum()),empirical_exact_commits=float(np.mean(empirical_exact)),
                     empirical_learned_commits=float(np.mean(empirical_learned)),unsafe_authorizations=int(np.sum(unsafe)),
                     monte_carlo_trials=256,exact_standard_error=float(np.std(empirical_exact,ddof=1)/16),
                     learned_standard_error=float(np.std(empirical_learned,ddof=1)/16))
            heterogeneous.append(row)
            path=ROOT/'controlled-progress'/f'{spec["label"]}-{state["id"]}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,q=q,p=p,available=available,formula=formula,exact=empirical_exact,learned=empirical_learned,unsafe=unsafe)
            print(f'controlled {spec["label"]} {state["id"]} predicted={formula.sum():.3f} exact={np.mean(empirical_exact):.3f}',flush=True)
    atomic_json(RESULTS/'controlled-progress.json',dict(cliques=clique,heterogeneous=heterogeneous))
