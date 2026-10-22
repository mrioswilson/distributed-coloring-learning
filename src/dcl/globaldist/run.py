"""Resumable atomic bank queue. No training or publication entry point."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,time,signal,sys
import numpy as np
import torch
from .core import *
from .sampling import sample
from dcl.study.storage import atomic_write,now,provenance
from dcl.localpalettes.runner import configure,load,persistent_gpu_lock

def read(p):return json.loads(Path(p).read_text())

def plan():
    freeze=read(OUT/'freeze.json')
    assert freeze['protocol_sha256']==file_hash('docs/GLOBAL_COLORING_DISTRIBUTION_PROTOCOL.md')
    gs=read(OUT/'graphs.json');ss=read(OUT/'checkpoints.json')
    assert freeze['graphs_sha256']==file_hash(OUT/'graphs.json')
    assert freeze['checkpoints_sha256']==file_hash(OUT/'checkpoints.json')
    tasks=[]
    for s in [None]+ss:
        label='Luby' if s is None else s['label']
        for g in gs:
            banks=[('confirmatory',250)]
            if g['stage']=='small':banks.append(('estimation',2000))
            if g['id'] in ('path3','star-d200','regular-n256-d100') and (s is None or s['seed']==41000):
                banks.extend([('power',2000),('classifier',300)])
            if s is None and g['id'] in ('path3','regular-n256-d100'):banks.append(('negative-luby',10000))
            if s is not None and g['id']=='path3':banks.append(('negative-model',6000))
            for stage,m in banks:
                ident=f'{stage}__{label}__{g["id"]}'
                tasks.append(dict(id=ident,stage=stage,m=m,model=s,graph=g['id'],status='pending'))
    # Fixed graph order means all models see identical graphs, no selection by results.
    return dict(tasks=tasks,gpu_seconds=freeze['pilot']['gpu_seconds'],active=None,created=now(),protocol_commit='80195c5',
                implementation=provenance(),budget_seconds=43200)

def budget_alarm(signum,frame):raise TimeoutError('GPU-process budget exhausted')

def main():
    configure();ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    with persistent_gpu_lock():
        state=read(ROOT/'queue.json') if (ROOT/'queue.json').exists() else plan()
        # Fail closed: an interrupted active interval includes elapsed downtime.
        if state.get('active'):
            task=next(t for t in state['tasks'] if t['id']==state['active'])
            receipt=ROOT/'receipts'/f"{task['id']}.json"
            if receipt.exists():
                rec=read(receipt);state['gpu_seconds']+=rec['seconds'];task['status']='complete';task.update(rec)
            else:
                state['gpu_seconds']+=max(0,time.time()-state['active_wall_start']);task['status']='pending'
            state['active']=None;atomic_json(ROOT/'queue.json',state)
        gs={g['id']:g for g in read(OUT/'graphs.json')};loaded=None;model=None
        session_start=time.monotonic();base=state['gpu_seconds']
        signal.signal(signal.SIGALRM,budget_alarm)
        for task in state['tasks']:
            if task['status']=='complete':continue
            state['gpu_seconds']=base+time.monotonic()-session_start
            left=state['budget_seconds']-state['gpu_seconds']
            if left<60:break
            label='Luby' if task['model'] is None else task['model']['label']
            state.update(active=task['id'],active_wall_start=time.time());task['status']='running'
            atomic_json(ROOT/'queue.json',state);begin=time.monotonic()
            signal.alarm(max(1,int(left)))
            try:
                if label!=loaded:
                    if model is not None:del model;model=None;torch.cuda.empty_cache()
                    model=None if task['model'] is None else load(task['model']);loaded=label
                    if model is not None:assert not model.training
                a=sample(gs[task['graph']],model,label,task['stage'],task['m'])
                path=ROOT/'banks'/f"{task['id']}.npz"
                atomic_write(path,lambda f:np.savez_compressed(f,**a))
                rec=dict(file=str(path),sha256=file_hash(path),seconds=time.monotonic()-begin,
                         statuses=np.bincount(a['status'],minlength=4).tolist(),completed=now())
                atomic_json(ROOT/'receipts'/f"{task['id']}.json",rec)
                task.update(rec);task['status']='complete';state['active']=None
                print(json.dumps(dict(task=task['id'],seconds=rec['seconds'],statuses=rec['statuses'])),flush=True)
            except TimeoutError:
                task['status']='budget-interrupted';state['active']=None
                break
            finally:
                signal.alarm(0);state['gpu_seconds']=base+time.monotonic()-session_start
                atomic_json(ROOT/'queue.json',state)
        state['gpu_seconds']=base+time.monotonic()-session_start;state['active']=None
        atomic_json(ROOT/'queue.json',state)
        atomic_json(OUT/'sampling-manifest.json',state)
        print(json.dumps(dict(complete=sum(t['status']=='complete' for t in state['tasks']),total=len(state['tasks']),gpu_hours=state['gpu_seconds']/3600)),flush=True)
if __name__=='__main__':main()
