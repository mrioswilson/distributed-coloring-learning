"""Exploratory independent replication of the largest progress residual.

Primary 256-trial measurements are NEVER replaced. Selection uses those results
and is explicitly post hoc; new RNG streams provide a fresh diagnostic check.
"""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from dcl.study.storage import atomic_json, file_hash, now
from dcl.universal.rules import available_matrix, exact_commit_labels
from .adapter import LocalPaletteAdapter, converted_uniform, choices, first_state
from .config import ROOT, RESULTS, seed
from .progress import expected_confirmations
from .runner import configure, load, persistent_gpu_lock, read, specs


def run():
    configure()
    with persistent_gpu_lock():
        queue=read(ROOT/'queue.json')
        if queue['gpu_seconds']>=12*3600:raise RuntimeError('GPU budget exhausted')
        start=time.monotonic()
        table=pd.read_csv(RESULTS/'heterogeneous-progress.csv')
        table['residual_z']=(table.empirical_exact_commits-table.expected_exact_commits_plugin)/table.exact_standard_error
        original=table.loc[table.residual_z.abs().idxmax()].to_dict()
        state=next(s for s in read(RESULTS/'autonomous-state-manifest.json') if s['id']==original['state'])
        spec=next(s for s in specs() if s['label']==original['model'])
        model=load(spec);adapter=LocalPaletteAdapter(model)
        colors=np.array(state['colors']);edges=np.array(state['edges']);palette=np.array(state['palettes'])
        pending=np.flatnonzero(colors<0);n=len(colors);width=int(palette.max())
        available=available_matrix(colors,edges,palette)
        ct=torch.tensor(colors,device='cuda');et=torch.tensor(edges,device='cuda');pt=torch.tensor(palette,device='cuda')
        pseed=seed('posthoc-progress-replication','participation',spec['label'],state['id'])
        qseeds={int(v):seed('posthoc-progress-replication','candidate',spec['label'],state['id'],int(v)) for v in pending}
        q=np.zeros((n,width));q_counts=np.zeros((n,width),np.int64)
        with torch.inference_mode():
            rp=torch.tensor(converted_uniform(np.random.default_rng(pseed).random(1048576)),device='cuda')
            p_actual=float((model.participation(torch.stack((rp,torch.ones_like(rp)),-1)).squeeze(-1)>0).float().mean())
            for v in pending:
                h,a=first_state(model,ct,et,torch.tensor([v],device='cuda'),int(palette[v]))
                selected=choices(model,(h[0],a[0]),converted_uniform(np.random.default_rng(qseeds[int(v)]).random(16384)))
                q_counts[v,:palette[v]]=np.bincount(selected,minlength=int(palette[v]))
                q[v]=q_counts[v]/16384
        p=np.where(colors<0,p_actual,0.)
        prediction=float(expected_confirmations(q,p,colors,edges,available).sum())
        test_seed=seed('posthoc-progress-replication','test',spec['label'],state['id'])
        rng=np.random.default_rng(test_seed)
        observed=[];learned=[];hist=np.zeros_like(q_counts);parts=np.zeros(n,np.int64)
        for trial in range(1024):
            draws=converted_uniform(rng.random((n,2)))
            output=adapter(ct,torch.tensor(draws[:,0],device='cuda'),torch.tensor(draws[:,1],device='cuda'),et,pt)
            b=output.participate.cpu().numpy();c=output.candidate.cpu().numpy()
            exact=exact_commit_labels(colors,edges,b,c)
            accept=(output.commit_logit>0).cpu().numpy()&b&(colors<0)
            observed.append(int(exact.sum()));learned.append(int(accept.sum()))
            np.add.at(hist,(pending,c[pending]),1);parts+=b
            if trial%128==127:print(f'fresh progress {trial+1}/1024, mean={np.mean(observed):.4f}',flush=True)
        observed=np.array(observed);se=float(observed.std(ddof=1)/np.sqrt(len(observed)))
        out=ROOT/'posthoc-progress-replication.npz'
        np.savez_compressed(out,q=q,p=p,q_counts=q_counts,exact=observed,learned=learned,candidate_counts=hist,participations=parts)
        seconds=time.monotonic()-start
        result=dict(kind='post-hoc-independent-diagnostic',selection='largest absolute original residual / empirical standard error, ignoring prediction uncertainty',
                    original=original,model=spec['label'],state=state['id'],checkpoint_sha256=spec['sha256'],
                    prediction=prediction,observed_mean=float(observed.mean()),observed_standard_error=se,
                    residual_z_observation_only=float((observed.mean()-prediction)/se),
                    learned_exact_counts_equal=bool(np.array_equal(observed,np.array(learned))),
                    participation_estimate=p_actual,participation_draws=1048576,candidate_draws_per_node=16384,trials=1024,
                    seeds=dict(participation=pseed,test=test_seed,candidate=qseeds),
                    arrays=str(out),arrays_sha256=file_hash(out),seconds=seconds,finished=now(),
                    uncertainty_note='Prediction uses independently estimated marginals; displayed SE covers observations only. Original primary results retained unchanged.')
        atomic_json(RESULTS/'progress-replication.json',result)
        queue['gpu_seconds']+=seconds
        queue.setdefault('posthoc_gpu_checks',[]).append(dict(id='progress-replication',seconds=seconds,result='results/ucgnn-local-palettes/progress-replication.json'))
        atomic_json(ROOT/'queue.json',queue);atomic_json(RESULTS/'execution-manifest.json',queue)
        print(json.dumps({k:v for k,v in result.items() if k not in ('seeds','original')}),flush=True)


if __name__=='__main__':run()
