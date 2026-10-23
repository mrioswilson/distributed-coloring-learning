"""Read-only trajectory observers and common frozen teacher-state probes.

No observer output enters an action. Candidate-grid masses are numerical
quadrature diagnostics, not exact probabilities or causal effect estimates.
"""
import argparse
import time
import signal
import numpy as np
import torch
from .common import *
from .inference import measurements
from .run import BudgetExceeded,alarm
from dcl.globaldist.core import seed as run_seed
from dcl.globaldist.sampling import sample
from dcl.localpalettes.adapter import LocalPaletteAdapter,converted_uniform,first_state,choices
from dcl.localpalettes.runner import configure,load,persistent_gpu_lock
from dcl.universal.rules import teacher_offer,exact_commit_labels,available_matrix

def trace(g,model,label,m=8):
    n=g['n'];e=np.asarray(g['edges'],np.int64).reshape(2,-1);pal=np.array(g['degrees'])+1;K=g['delta']+1
    ee=np.concatenate([e+j*n for j in range(m)],axis=1);pp=np.tile(pal,m)
    seeds=[run_seed('factorial-v1/mechanism',g['id'],label,'run',i) for i in range(m)]
    rr=[np.random.default_rng(s) for s in seeds];c=np.full((m,n),-1,np.int64)
    adapter=None if model is None else LocalPaletteAdapter(model)
    if model is not None:
        device=next(model.parameters()).device;et=torch.as_tensor(ee,device=device);pt=torch.as_tensor(pp,device=device)
    states=[];cand_counts=[];part_counts=[];commits=[];reference=[];bad=[]
    node_candidates=[];node_participate=[];node_commit=[];node_available=[]
    for r in range(128):
        if not (c<0).any():break
        states.append(c.copy().astype(np.int16))
        u=np.stack([a.random((n,2)) for a in rr]);rp=u[:,:,0].ravel();rc=u[:,:,1].ravel();flat=c.ravel()
        if model is None:
            cand,_=teacher_offer(flat,rc,ee,pp);part=(rp<.5)&(flat<0);commit=exact_commit_labels(flat,ee,part,cand)
        else:
            with torch.inference_mode():
                q=adapter(torch.as_tensor(flat,device=device),torch.as_tensor(converted_uniform(rp),device=device),
                          torch.as_tensor(converted_uniform(rc),device=device),et,pt)
            cand=q.candidate.cpu().numpy();part=q.participate.cpu().numpy()
            commit=(q.commit_logit>0).cpu().numpy()&part&(flat<0)
        # Decisions above are already fixed. The following are observers only.
        available=available_matrix(flat,ee,pp);avail_n=available.sum(1)
        node_candidates.append(cand.reshape(m,n).astype(np.int16).copy())
        node_participate.append(part.reshape(m,n).copy())
        node_commit.append(commit.reshape(m,n).copy())
        node_available.append(avail_n.reshape(m,n).astype(np.int16).copy())
        expected=available/np.maximum(avail_n[:,None],1)*(flat<0)[:,None]
        reference.append(expected.reshape(m,n,K).sum(1))
        def histogram(mask):
            v=cand.reshape(m,n);b=mask.reshape(m,n)
            return np.array([np.bincount(row[ok],minlength=K) for row,ok in zip(v,b)],np.int32)
        cand_counts.append(histogram((flat<0)&(cand>=0)))
        part_counts.append(histogram(part&(cand>=0)))
        commits.append(histogram(commit&(cand>=0)))
        safe=exact_commit_labels(flat,ee,part,cand)
        forbidden=np.zeros(len(flat),bool);ix=np.flatnonzero(cand>=0)
        forbidden[ix]=~available[ix,cand[ix]]
        bad.append(np.c_[forbidden.reshape(m,n).sum(1),(commit&~safe).reshape(m,n).sum(1)])
        flat[commit]=cand[commit]
    status=np.array([2 if (x<0).any() else (0 if (x<pal).all() and (x[e[0]]!=x[e[1]]).all() else 1) for x in c],np.int8)
    return dict(colors=c.astype(np.int16),status=status,seeds=np.array(seeds,np.uint64),
                states=np.array(states),candidate_counts=np.array(cand_counts),participating_counts=np.array(part_counts),
                commit_counts=np.array(commits),same_state_teacher_expected=np.array(reference),observer_errors=np.array(bad),
                node_candidates=np.array(node_candidates),node_participate=np.array(node_participate),
                node_commit=np.array(node_commit),node_available_counts=np.array(node_available))

def quadrature(g,state,model,count):
    pending=np.flatnonzero(state<0)
    if not len(pending):return None
    v=int(pending[0]);device=next(model.parameters()).device;e=np.array(g['edges']);k=g['degrees'][v]+1
    h,a=first_state(model,torch.as_tensor(state,device=device,dtype=torch.long),torch.as_tensor(e,device=device,dtype=torch.long),torch.tensor([v],device=device,dtype=torch.long),k)
    grid=converted_uniform((np.arange(count)+.5)/count)
    selected=choices(model,(h[0],a[0]),grid)
    masses=np.bincount(selected,minlength=k)/count
    available=available_matrix(state,e,np.array(g['degrees'])+1)[v,:k];target=available/available.sum()
    return dict(vertex=v,grid_count=count,q=masses.tolist(),teacher=target.tolist(),available=available.tolist(),
                candidate_tv_grid=float(.5*abs(masses-target).sum()),q_top=float(masses[-1]),teacher_q_top=float(target[-1]),
                q_highest_available=float(masses[np.flatnonzero(available)[-1]]),forbidden_mass_grid=float(masses[~available].sum()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');args=p.parse_args()
    f=verify_frozen();begin=time.monotonic();ledger=ROOT/'budget.json';manifest=ROOT/'mechanism-queue.json'
    gs=[g for g in graphs() if (g['stage']=='main' and g['rep']==0) or g['stage']=='historical-replication']
    with persistent_gpu_lock():
        budget=read(ledger)
        if budget.get('active_wall'):
            budget['seconds']+=max(0.,time.time()-budget['active_wall']);budget['active_wall']=None
        if budget['seconds']>=budget['limit']-60:return
        configure();budget['seconds']+=time.monotonic()-begin
        budget.setdefault('initialization_charges',[]).append(dict(phase='mechanism',seconds=time.monotonic()-begin))
        tasks=[dict(graph=g['id'],model=s,id=f'mechanism__{"Luby" if s is None else s["label"]}__{g["id"]}') for s in [None]+specs() for g in gs]
        state=read(manifest) if manifest.exists() else dict(tasks=tasks,complete=[],created=now())
        loaded=None;model=None
        for t in state['tasks']:
            if t['id'] in state['complete']:continue
            left=budget['limit']-budget['seconds']
            if left<60:break
            started=time.monotonic();budget['active_wall']=time.time();atomic_json(ledger,budget)
            signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(left-30)))
            try:
                g=next(g for g in gs if g['id']==t['graph']);label='Luby' if t['model'] is None else t['model']['label']
                if label!=loaded:
                    if model is not None:del model;torch.cuda.empty_cache()
                    model=None if t['model'] is None else load(t['model']);loaded=label
                path=ROOT/'mechanism'/f'{t["id"]}.npz'
                if not path.exists():save_bank(path,**trace(g,model,label))
                if model is not None:
                    with np.load(ROOT/'mechanism'/f'mechanism__Luby__{g["id"]}.npz') as a:
                        states=a['states'][:,0]
                    result=[]
                    for round_id in (0,2,4,8):
                        if round_id>=len(states):continue
                        coarse=quadrature(g,states[round_id],model,1024)
                        fine=quadrature(g,states[round_id],model,4096)
                        if fine is not None:
                            result.append(dict(round=round_id,coarse_q_top=coarse['q_top'],
                                               coarse_to_fine_l1=float(np.abs(np.array(coarse['q'])-fine['q']).sum()),**fine))
                    atomic_json(OUT/'candidate-probes'/f'{g["id"]}__{label}.json',dict(graph=g['id'],model=label,n=g['n'],delta=g['delta'],probes=result))
                t.update(seconds=time.monotonic()-started,sha256=file_hash(path),
                         source_sha256=file_hash(__file__),status='complete')
                state['complete'].append(t['id']);atomic_json(manifest,state)
                print(json.dumps(dict(id=t['id'],seconds=time.monotonic()-started)),flush=True)
            except BudgetExceeded:break
            finally:
                signal.alarm(0);budget['seconds']+=time.monotonic()-started;budget['active_wall']=None;atomic_json(ledger,budget)
        print(json.dumps(dict(complete=len(state['complete']),total=len(state['tasks']),gpu_hours=budget['seconds']/3600)),flush=True)

if __name__=='__main__':main()
