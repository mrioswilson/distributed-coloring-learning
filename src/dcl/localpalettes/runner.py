"""Persistent, single-GPU, atomic-per-case coordinator. No training commands."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from dcl.paperstudy.training import load_model
from dcl.study.storage import atomic_json, file_hash, now, provenance, resources, training_lock
from .adapter import LocalPaletteAdapter, choices, converted_uniform, first_state
from .cohort import build_graphs, build_states, edge_array, make_graph, palette_for, star_edges
from .config import BUDGET_SECONDS, PROTOCOL, REGIMES, RESULTS, ROOT, seed
from .evaluation import center_cache, evaluate_law, rollout


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.65)


def specs():
    return read(RESULTS/'frozen-checkpoints.json')


def load(spec):
    if file_hash(spec['checkpoint'])!=spec['sha256']:raise RuntimeError('Checkpoint hash changed')
    if spec.get('kind')=='historical-e1':
        from dcl.universal.train import load_model as old_load
        model,_=old_load(spec['checkpoint'],'cuda')
    else:model,_=load_model(spec['checkpoint'],'cuda')
    return model


def protocol_hash():
    expected=read(RESULTS/'preregistration-environment.json')['protocol_sha256']
    actual=file_hash(PROTOCOL)
    if actual!=expected:raise RuntimeError('Frozen protocol changed; add a prospective addendum instead')
    return actual


@contextmanager
def persistent_gpu_lock():
    """Pass this flock descriptor to workers, retaining exclusion if parent dies."""
    path=Path.home()/'.cache/dcl-m6-gpu.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another GPU study is active; no concurrent launch') from None
        yield handle.fileno()


def recover_and_extend(state):
    for task in state['tasks']:
        if task['status']=='running':
            # A surviving child retains the flock, so this point is reached
            # only once it has exited. Conservatively charge lost bookkeeping.
            receipt=ROOT/'task-receipts'/f'{task["id"]}.json'
            if receipt.exists():
                charge=read(receipt)['seconds'];task['status']='complete'
            else:
                wall=max(0.,(datetime.now(timezone.utc)-datetime.fromisoformat(task['started'])).total_seconds())
                charge=min(1800.,wall);task['status']='pending'
            state['gpu_seconds']+=charge
            task['recovery_charged_seconds']=charge
    supplement=RESULTS/'historical-e1-checkpoints.json'
    if supplement.exists():
        existing={t['id'] for t in state['tasks']}
        for spec in read(supplement):
            ident=f'laws-{spec["label"]}'
            if ident not in existing:state['tasks'].append(dict(id=ident,kind='laws',model=spec,status='pending',secondary=True))
    return state


def status():
    state=read(ROOT/'queue.json') if (ROOT/'queue.json').exists() else dict(tasks=[],gpu_seconds=0)
    counts=Counter(t['status'] for t in state['tasks'])
    print(json.dumps(dict(tasks=counts,gpu_process_hours=state['gpu_seconds']/3600,budget_hours=12,
                         active=state.get('active'),pilot=(RESULTS/'pilot.json').exists()),indent=2))
    return state


def update_status(state):
    counts=Counter(t['status'] for t in state['tasks'])
    complete=bool(state['tasks']) and all(t['status']=='complete' for t in state['tasks'])
    execution='Completed; no live coordinator is required. Resume is idempotent.' if complete else 'Running/resumable coordinator in tmux ucgnn-local-palette-study.'
    text=f'''# Local-palette study status

Updated {now()}; branch research/ucgnn-local-palettes; protocol commit f516530.
No new training. Historical data untouched. No remote push.

GPU-process hours: {state['gpu_seconds']/3600:.4f} / 12.
Tasks: {dict(counts)}. Active: {state.get('active')}.

See [LOCAL_PALETTES.md](LOCAL_PALETTES.md) for commands and scientific scope.
Runtime manifest: data/ucgnn-local-palettes/queue.json.
Per-case outputs: data/ucgnn-local-palettes/graphs/ and distributions/.
Reports: results/ucgnn-local-palettes/.
Execution: {execution}
Explicit resume is required after reboot; no reboot persistence is promised.
'''
    Path('STATUS.md').write_text(text)


def pilot():
    configure();protocol_hash()
    rows=[];invariance=[];timings=[]
    start=time.monotonic()
    with training_lock(ROOT):
        for spec in [s for s in specs() if s['seed']==41000]:
            model=load(spec)
            for family,d in (('REGULAR',3),('IRREGULAR',3),('STAR',15),('HETEROGENEOUS',15),('BIPARTITE',15),('LOW_HIGH',15)):
                g,_=make_graph(family,d,0,pilot=True)
                edges=edge_array(g);degrees=np.array([g.degree(i) for i in g])
                ident=f'pilot-{family}-{d}'
                result,colors,capture=rollout(model,edges,degrees,'LOCAL',ident,0,rounds=64,capture_round=1)
                rows.append(dict(model=spec['label'],family=family,**{k:v for k,v in result.items() if k!='trace'}))
            # Actual trained weights, same mixed legal state, original vs adapter.
            colors=torch.tensor([-1,5,-1,0,-1,1,-1],device='cuda')
            palette=torch.tensor([3,6,2,4,7,2,1],device='cuda')
            edges=torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]],device='cuda')
            rp=torch.full((7,),.3,device='cuda');rc=torch.linspace(.07,.93,7,device='cuda')
            with torch.inference_mode():
                old=model(colors,rp,rc,edges,palette)
                new=LocalPaletteAdapter(model)(colors,rp,rc,edges,palette,output_width=301)
            valid=torch.arange(7,device='cuda')[None,:]<palette[:,None]
            delta=float((old.candidate_logits[valid]-new.candidate_logits[:,:7][valid]).abs().max())
            cdelta=float((old.commit_logit-new.commit_logit).abs().max())
            passed=torch.equal(old.candidate,new.candidate) and torch.equal(old.commit_logit>0,new.commit_logit>0)
            invariance.append(dict(model=spec['label'],max_score_error=delta,max_commit_error=cdelta,hard_actions_equal=passed))
            if not passed:raise RuntimeError(f'Historical action equivalence failed {spec["label"]}')
            for k in (51,201,501):
                s=dict(k=k,colors=[-1]*k)
                with torch.inference_mode():
                    cache=center_cache(model,s)
                    torch.cuda.synchronize();t=time.monotonic()
                    output=choices(model,cache,converted_uniform(np.random.default_rng(seed('pilot-speed',k)).random(4096)))
                    torch.cuda.synchronize()
                timings.append(dict(model=spec['label'],k=k,draws=len(output),seconds=time.monotonic()-t))
            del model;torch.cuda.empty_cache()
    elapsed=time.monotonic()-start
    result=dict(passed=True,interpretation='Execution/invariance validation; model failures are retained, not promotion exclusions',
                graph_cases=rows,invariance=invariance,candidate_timings=timings,gpu_seconds=elapsed,
                protocol_sha256=protocol_hash(),environment=provenance(),created=now())
    atomic_json(RESULTS/'pilot.json',result)
    print(json.dumps(dict(pilot='passed',cases=len(rows),statuses=Counter(x['status'] for x in rows),seconds=elapsed)))


def plan():
    if not (RESULTS/'pilot.json').exists() or not read(RESULTS/'pilot.json')['passed']:
        raise RuntimeError('Verified pilot required')
    graphs=build_graphs();states=build_states()
    tasks=[]
    for s in [dict(label='exact-teacher',treatment='teacher',seed=None)]+specs():
        for regime in REGIMES:
            tasks.append(dict(id=f'graphs-{s["label"]}-{regime}',kind='graphs',model=s,regime=regime,status='pending'))
    for s in specs():tasks.append(dict(id=f'laws-{s["label"]}',kind='laws',model=s,status='pending'))
    tasks.append(dict(id='controlled-progress',kind='progress',status='pending'))
    state=dict(tasks=tasks,gpu_seconds=read(RESULTS/'pilot.json')['gpu_seconds'],active=None,
               protocol_sha256=protocol_hash(),created=now(),environment=provenance(),
               graph_specifications=len(graphs),conditional_states=len(states))
    atomic_json(ROOT/'queue.json',state)
    atomic_json(RESULTS/'task-plan.json',state)
    return state


def frozen_autonomous_states():
    directory=ROOT/'whole-states'
    states=[]
    if not directory.exists():return states
    for path in sorted(directory.glob('*.npz')):
        x=np.load(path)
        colors=x['colors'];pal=x['palette'];edges=x['edges']
        pending=np.flatnonzero(colors<0)
        if not len(pending):continue
        center=int(pending[0]);k=int(pal[center])
        from dcl.universal.rules import available_matrix
        a=np.flatnonzero(available_matrix(colors,edges,pal)[center,:k])
        if not len(a):continue
        states.append(dict(id=path.stem,k=k,available=a.tolist(),colors=colors.tolist(),palettes=pal.tolist(),
                           edges=edges.tolist(),center=center,pattern='AUTONOMOUS',kind='new-local-rollout',b3=False,
                           provenance=read(path.with_suffix('.json')),state_sha256=file_hash(path)))
    atomic_json(RESULTS/'autonomous-state-manifest.json',states)
    return states


def graph_worker(task):
    model=None if task['model']['label']=='exact-teacher' else load(task['model'])
    capture_ids=[g['id'] for g in sorted(build_graphs(),key=lambda g:g['id']) if g['family']=='HETEROGENEOUS'][:2]
    directory=ROOT/'graphs'/task['model']['label']/task['regime'];directory.mkdir(parents=True,exist_ok=True)
    for graph in build_graphs():
        x=np.load(graph['file']);edges,degrees=x['edges'],x['degree']
        for tape in (0,1):
            path=directory/f'{graph["id"]}-t{tape}.json'
            if path.exists():continue
            capture=(task['regime']=='LOCAL' and tape==0 and graph['id'] in capture_ids and
                     task['model']['label'] in ('normalized-s41000','deepsets-s41000','transformer-s41000'))
            result,colors,partial=rollout(model,edges,degrees,task['regime'],graph['id'],tape,capture_round=1 if capture else None)
            result.update(model=task['model']['label'],treatment=task['model']['treatment'],network_seed=task['model']['seed'],
                          graph=graph,regime=task['regime'],tape=tape,protocol_sha256=protocol_hash())
            atomic_json(path,result)
            if result['status']!='success':np.savez_compressed(path.with_suffix('.npz'),colors=colors,palette=palette_for(task['regime'],degrees))
            if capture and partial is not None:
                name=f'{task["model"]["label"]}-{graph["id"]}-t0-r1'
                dest=ROOT/'whole-states'/f'{name}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,colors=partial,edges=edges,palette=degrees+1)
                atomic_json(dest.with_suffix('.json'),dict(model=task['model']['label'],graph=graph['id'],family=graph['family'],
                            degree=graph['degree'],n=graph['n'],tape=tape,tape_seed=result['tape_seed'],round=1,regime='LOCAL'))
            print(f'{task["id"]} {graph["id"]} t{tape} {result["status"]} r={result["rounds"]}',flush=True)
            atomic_json(ROOT/'heartbeat.json',dict(task=task['id'],updated=now(),case=str(path)))


def laws_worker(task):
    model=load(task['model'])
    directory=ROOT/'distributions'/task['model']['label'];directory.mkdir(parents=True,exist_ok=True)
    historical=read(RESULTS/'historical-state-manifest.json') if (RESULTS/'historical-state-manifest.json').exists() else []
    for state in build_states()+frozen_autonomous_states()+historical:
        path=directory/f'{state["id"]}.json'
        if path.exists():continue
        with torch.inference_mode():result=evaluate_law(model,task['model']['label'],state,directory)
        atomic_json(path,result)
        print(f'{task["id"]} {state["id"]} TV={result["metrics"]["tv"]:.4f}',flush=True)
        atomic_json(ROOT/'heartbeat.json',dict(task=task['id'],updated=now(),case=str(path)))


def worker(task_id):
    configure();protocol_hash()
    task=next(t for t in read(ROOT/'queue.json')['tasks'] if t['id']==task_id)
    before=resources(ROOT,device='cuda');torch.cuda.reset_peak_memory_stats()
    source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                files={str(p):file_hash(p) for p in sorted(Path('src/dcl/localpalettes').glob('*.py'))})
    t=time.monotonic()
    if task['kind']=='graphs':graph_worker(task)
    elif task['kind']=='laws':laws_worker(task)
    elif task['kind']=='progress':
        from .progress import run_progress
        run_progress()
    atomic_json(ROOT/'task-receipts'/f'{task_id}.json',dict(task=task_id,finished=now(),seconds=time.monotonic()-t,
                before=before,peak_gpu_bytes=torch.cuda.max_memory_allocated(),reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
                protocol_sha256=protocol_hash(),source=source))


def run():
    configure();protocol_hash()
    with persistent_gpu_lock() as lock_fd:
        state=read(ROOT/'queue.json') if (ROOT/'queue.json').exists() else plan()
        state=recover_and_extend(state)
        for task in state['tasks']:
            if task['status']!='pending':continue
            if state['gpu_seconds']>=BUDGET_SECONDS:break
            resources(ROOT,device='cuda')
            state['active']=task['id'];task.update(status='running',started=now())
            atomic_json(ROOT/'queue.json',state);update_status(state)
            log=ROOT/'logs'/f'{task["id"]}.log';log.parent.mkdir(parents=True,exist_ok=True)
            start=time.monotonic()
            remaining=max(1,int(BUDGET_SECONDS-state['gpu_seconds']))
            with log.open('a') as handle:
                try:
                    result=subprocess.run([sys.executable,'-m','dcl.localpalettes','worker',task['id']],stdout=handle,stderr=subprocess.STDOUT,
                                          timeout=min(1800,remaining),pass_fds=(lock_fd,))
                    code=result.returncode
                except subprocess.TimeoutExpired:code=124
            elapsed=time.monotonic()-start
            state['gpu_seconds']+=elapsed
            receipt=ROOT/'task-receipts'/f'{task["id"]}.json'
            task.update(status='complete' if code==0 and receipt.exists() else 'technical_failure',returncode=code,seconds=elapsed,finished=now())
            state['active']=None
            atomic_json(ROOT/'queue.json',state);update_status(state)
            print(f'{task["id"]}: {task["status"]} ({elapsed:.1f}s), cumulative {state["gpu_seconds"]/3600:.3f}h',flush=True)
        atomic_json(RESULTS/'execution-manifest.json',state)
        update_status(state)
        from .report import report
        report()


def audit():
    protocol_hash()
    all_specs=specs()+(read(RESULTS/'historical-e1-checkpoints.json') if (RESULTS/'historical-e1-checkpoints.json').exists() else [])
    missing=[s['checkpoint'] for s in all_specs if not Path(s['checkpoint']).exists() or file_hash(s['checkpoint'])!=s['sha256']]
    state=status()
    result=dict(protocol_unchanged=True,checkpoints_unchanged=not missing,checkpoints_checked=len(all_specs),invalid_checkpoints=missing,
                graph_cases=len(list((ROOT/'graphs').rglob('*.json'))),conditional_laws=len(list((ROOT/'distributions').rglob('*.json'))),
                tasks_complete=sum(t['status']=='complete' for t in state['tasks']),tasks_total=len(state['tasks']),
                gpu_hours=state['gpu_seconds']/3600,environment=provenance())
    atomic_json(RESULTS/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='environment'},indent=2))
    if state['tasks'] and all(t['status']=='complete' for t in state['tasks']):
        from .completion_audit import run as complete_audit
        complete_audit()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('status','run','resume','audit','report','pilot','worker'))
    parser.add_argument('task',nargs='?')
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    if args.command=='status':status()
    elif args.command=='pilot':pilot()
    elif args.command in ('run','resume'):run()
    elif args.command=='worker':worker(args.task)
    elif args.command=='audit':audit()
    else:
        state=read(ROOT/'queue.json') if (ROOT/'queue.json').exists() else dict(tasks=[])
        if state['tasks'] and all(t['status']=='complete' for t in state['tasks']):
            from .finalize import finalize
            finalize()
        else:
            from .report import report
            report()


if __name__=='__main__':main()
