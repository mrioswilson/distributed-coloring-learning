"""Independent persistent queues sharing a single conservatively charged cap."""
import argparse,time,signal
import numpy as np
import torch
from .common import *
from dcl.globaldist.sampling import sample
from dcl.localpalettes.runner import configure,load,persistent_gpu_lock
from dcl.study.storage import now,provenance

def tasks(phase):
    gs={g['id']:g for g in graphs()};ss=specs();out=[]
    if phase=='cost':
        cells=[('path3',32),('star-d200',32),('regular-n256-d100',16),('regular-n512-d200',16),('clique-d200',8)]
        for s in [None]+[s for s in ss if s['seed']==41000]:
            for g,m in cells:out.append(dict(graph=g,m=m,model=s))
    elif phase in ('null','null2'):
        for s in [None]+[s for s in ss if s['seed']==41000]:out.append(dict(graph='path3',m=200000,model=s))
    elif phase=='confirmatory':
        freeze=read(OUT/'freeze.json');assert freeze['protocol_sha256']==file_hash('docs/ROBUST_GLOBAL_DISTRIBUTION_PROTOCOL.md')
        assert read(OUT/'instrument-acceptance.json')['passed']
        for name,digest in freeze['source_hashes'].items():assert file_hash(name)==digest,name
        for s in [None]+ss:
            for g in graphs():out.append(dict(graph=g['id'],m=2250 if g['id'] in freeze['extended_graphs'] else 750,model=s))
    else:raise ValueError(phase)
    for t in out:
        label='Luby' if t['model'] is None else t['model']['label']
        t.update(id=f'{phase}__{label}__{t["graph"]}',status='pending',stage=f'robust-v1/{phase}')
    return out

class BudgetExceeded(BaseException):
    """Must bypass the reused sampler's per-batch Exception handler."""
def alarm(signum,frame):raise BudgetExceeded('12-hour GPU-process cap')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['cost','null','null2','confirmatory']);args=parser.parse_args()
    process_begin=time.monotonic();DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    queue=DATA/f'queue-{args.phase}.json';ledger=DATA/'budget.json'
    with persistent_gpu_lock():
        state=read(queue) if queue.exists() else dict(phase=args.phase,tasks=tasks(args.phase),created=now(),active=None)
        budget=read(ledger) if ledger.exists() else dict(seconds=0.,limit=43200.,active_wall=None,provenance=provenance())
        if budget.get('active_wall'):
            budget['seconds']+=max(0.,time.time()-budget['active_wall']);budget['active_wall']=None
        if budget['limit']-budget['seconds']<60:
            atomic_json(ledger,budget);print('Budget unavailable; no GPU initialization',flush=True);return
        configure();initialization=time.monotonic()-process_begin;budget['seconds']+=initialization
        budget.setdefault('initialization_charges',[]).append(dict(phase=args.phase,seconds=initialization))
        atomic_json(ledger,budget);gs={g['id']:g for g in graphs()};model=None;loaded=None
        for t in state['tasks']:
            if t['status']=='complete':continue
            recpath=DATA/'receipts'/f"{t['id']}.json"
            if recpath.exists():t.update(read(recpath));t['status']='complete';continue
            left=budget['limit']-budget['seconds']
            if left<60:break
            label='Luby' if t['model'] is None else t['model']['label'];begin=time.monotonic()
            budget['active_wall']=time.time();state['active']=t['id'];atomic_json(ledger,budget);atomic_json(queue,state)
            signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(left-30)))
            try:
                if loaded!=label:
                    if model is not None:del model;torch.cuda.empty_cache()
                    model=None if t['model'] is None else load(t['model']);loaded=label
                a=sample(gs[t['graph']],model,label,t['stage'],t['m'])
                file=DATA/'banks'/f"{t['id']}.npz";save_bank(file,**a)
                rec=dict(file=str(file),sha256=file_hash(file),seconds=time.monotonic()-begin,statuses=np.bincount(a['status'],minlength=4).tolist(),completed=now())
                atomic_json(recpath,rec);t.update(rec);t['status']='complete'
                print(json.dumps(dict(id=t['id'],seconds=rec['seconds'],statuses=rec['statuses'])),flush=True)
            except BudgetExceeded:t['status']='budget_interrupted';break
            finally:
                signal.alarm(0);budget['seconds']+=time.monotonic()-begin;budget['active_wall']=None;state['active']=None
                atomic_json(ledger,budget);atomic_json(queue,state)
        atomic_json(queue,state)
        print(json.dumps(dict(phase=args.phase,complete=sum(t['status']=='complete' for t in state['tasks']),total=len(state['tasks']),gpu_hours=budget['seconds']/3600)),flush=True)
if __name__=='__main__':main()
