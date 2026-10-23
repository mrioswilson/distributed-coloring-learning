"""Atomic banks, independent per-run seeds, shared lock, hard GPU-process cap."""
import argparse
import signal
import time
import numpy as np
import torch
from .common import *
from dcl.globaldist.sampling import sample
from dcl.localpalettes.runner import configure, load, persistent_gpu_lock

class BudgetExceeded(BaseException):
    pass

def alarm(signum,frame):
    raise BudgetExceeded('12 GPU-process hours including pilots')

def tasks(phase):
    rows=read(OUT/('cost-graphs.json' if phase=='cost' else 'pilot-graphs.json')) if phase!='confirmation' else graphs()
    models=[None]+([s for s in specs() if s['seed']==41000] if phase!='confirmation' else specs())
    m=(32 if phase=='pilot' else 128) if phase!='confirmation' else verify_frozen()['total_runs_per_bank']
    out=[]
    strata=['main','historical-replication','secondary'] if phase=='confirmation' else [None]
    for stratum in strata:
        for s in models:
            label='Luby' if s is None else s['label']
            for g in rows:
                if stratum is not None and g['stage']!=stratum:continue
                out.append(dict(id=f'{phase}__{label}__{g["id"]}',graph=g['id'],model=s,m=m,
                                status='pending',stage=f'factorial-v1/{phase}'))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','cost','confirmation']);args=p.parse_args()
    process_begin=time.monotonic();ROOT.mkdir(parents=True,exist_ok=True)
    queue=ROOT/f'queue-{args.phase}.json';ledger=ROOT/'budget.json'
    with persistent_gpu_lock():
        if args.phase=='confirmation': verify_frozen()
        state=read(queue) if queue.exists() else dict(phase=args.phase,tasks=tasks(args.phase),created=now(),active=None)
        budget=read(ledger) if ledger.exists() else dict(seconds=0.,limit=43200.,active_wall=None)
        if budget.get('active_wall'):
            budget['seconds']+=max(0.,time.time()-budget['active_wall']);budget['active_wall']=None
        if budget['seconds']>=budget['limit']-60: return
        configure();budget['seconds']+=time.monotonic()-process_begin
        budget.setdefault('initialization_charges',[]).append(dict(phase=args.phase,seconds=time.monotonic()-process_begin))
        gs={g['id']:g for g in read(OUT/'pilot-graphs.json')+graphs()+ (read(OUT/'cost-graphs.json') if (OUT/'cost-graphs.json').exists() else [])}
        model=None;loaded=None
        atomic_json(ledger,budget)
        for t in state['tasks']:
            if t['status']=='complete':continue
            receipt=ROOT/'receipts'/f'{t["id"]}.json'
            if receipt.exists():t.update(read(receipt));t['status']='complete';continue
            left=budget['limit']-budget['seconds']
            reserve=read(OUT/'freeze.json').get('mechanism_reserve_seconds',600) if args.phase=='confirmation' else 0
            if left < reserve+60:break
            label='Luby' if t['model'] is None else t['model']['label']
            begin=time.monotonic();budget['active_wall']=time.time();state['active']=t['id']
            atomic_json(ledger,budget);atomic_json(queue,state)
            signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(left-reserve-30)))
            try:
                if loaded!=label:
                    if model is not None:del model;torch.cuda.empty_cache()
                    model=None if t['model'] is None else load(t['model']);loaded=label
                a=sample(gs[t['graph']],model,label,t['stage'],t['m'],batch_nodes=8192)
                path=ROOT/'banks'/f'{t["id"]}.npz';save_bank(path,**a)
                rec=dict(file=str(path),sha256=file_hash(path),seconds=time.monotonic()-begin,
                         statuses=np.bincount(a['status'],minlength=4).tolist(),completed=now())
                atomic_json(receipt,rec);t.update(rec);t['status']='complete'
                print(json.dumps(dict(id=t['id'],seconds=rec['seconds'],statuses=rec['statuses'])),flush=True)
            except BudgetExceeded:
                t['status']='budget_interrupted';break
            finally:
                signal.alarm(0);budget['seconds']+=time.monotonic()-begin;budget['active_wall']=None;state['active']=None
                atomic_json(ledger,budget);atomic_json(queue,state)
        print(json.dumps(dict(phase=args.phase,complete=sum(t['status']=='complete' for t in state['tasks']),total=len(state['tasks']),gpu_hours=budget['seconds']/3600)),flush=True)

if __name__=='__main__':main()
