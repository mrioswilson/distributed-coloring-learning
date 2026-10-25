#!/usr/bin/env python3
"""Replay a full original inference batch; preserve every diagnostic step."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from dcl.degree.adapter import FrozenAdapter
from dcl.degree.evaluation import combine, split_decision, component_diagnostics
from dcl.degree.inventory import ROOT
from dcl.degree.run import suite_for
from dcl.degree.teachers import DegreeProgram, exact_choices, availability
from dcl.multiteacher.programs import DETECT, ASSIGN
from dcl.samplecomplexity.model import QueryPolicy
from dcl.study.storage import atomic_json, file_hash


def replay(cp, condition, plan):
    path=ROOT/'confirmatory'/cp['id']/condition['id']/'rows.json'
    expected=json.loads(path.read_text())
    bad=[i for i,r in enumerate(expected) if not r['success']]
    if not bad:return None
    index=bad[0]
    start=index//plan['batch']*plan['batch']
    subset=suite_for(plan,condition,cp['seed']%100-1)[start:start+plan['batch']]
    programs=[DegreeProgram(g,cp['teacher'],m['tape_seed'],condition['palette']) for g,m in subset]
    model=QueryPolicy(cp['teacher'],cp['architecture']).eval()
    model.load_state_dict(torch.load(cp['path'],weights_only=False)['model'])
    adapter=FrozenAdapter(model)
    frames=[]
    wall,cpu=time.perf_counter(),time.process_time()
    target=index-start
    teacher_context_valid=True
    with torch.inference_mode():
        while any(not p.finished for p in programs):
            groups=defaultdict(list)
            for i,p in enumerate(programs):
                if not p.finished:groups[p.phase,p.k].append((i,p,p.snapshot()))
            for group in groups.values():
                joined,s,offsets=combine([(p,s) for _,p,s in group])
                decisions=split_decision(adapter(joined,s),offsets)
                for (i,p,s),d in zip(group,decisions):
                    before=p.colors.copy()
                    if i==target:
                        exact=exact_choices(p,s)
                        free=availability(p,s)
                        roots=d.roots
                        written=d.choices.color[roots]>0
                        unsafe=written & ~free[roots,np.maximum(0,d.choices.color[roots]-1)] if s.phase!=DETECT else np.zeros(len(roots),bool)
                        components=component_diagnostics(p,s,d)
                        frame=dict(step=p.steps,phase=s.phase,teacher_context_valid=teacher_context_valid,
                            colors=s.colors.tolist(),permission=s.permission.tolist(),draws=s.draws.tolist(),
                            preferences=s.preferences.tolist(),roots=roots.tolist(),
                            participation=d.participation.tolist(),candidates=d.candidates.tolist(),
                            commit=d.commit.tolist(),commit_logits=d.commit_logits.tolist(),
                            availability=d.availability.tolist(),scores=d.scores.tolist(),
                            predicted_color=d.choices.color.tolist(),predicted_mark=d.choices.mark.tolist(),
                            exact_color=exact.color.tolist(),exact_mark=exact.mark.tolist(),
                            forbidden_writes=roots[unsafe].tolist(),component_disagreements={
                                k:int(np.sum(a!=b)) for k,(a,b) in components.items()})
                    p.apply(s,d.choices)
                    if i==target:
                        if cp['teacher']=='SPECULATIVE_GM':
                            a,b=p.edges
                            bad_edge=(p.colors[a]>=0)&(p.colors[a]==p.colors[b])
                            cross_slot=bad_edge&(p.slots[a]!=p.slots[b])
                            frame['cross_slot_conflict_edges']=int(np.sum(cross_slot & (a<b)))
                            frame['scheduled_write_omissions']=int(np.sum(~written)) if s.phase!=DETECT else 0
                            frame['uncovered_conflict_edges']=int(np.sum(bad_edge&(a<b)&~p.marks[a]&~p.marks[b])) if s.phase==DETECT and p.phase!=DETECT else None
                            if unsafe.any() or frame['scheduled_write_omissions'] or cross_slot.any():teacher_context_valid=False
                        frames.append(frame)
                    valid=np.all((p.colors>=-1)&(p.colors<p.palette)) and np.array_equal(before[~s.permission],p.colors[~s.permission])
                    if cp['teacher']!='SPECULATIVE_GM':
                        valid &= not p.bad_edges() and np.array_equal(before[before>=0],p.colors[before>=0])
                    if not valid:p.finished,p.failure=True,p.failure or 'invariant'
    p=programs[target]
    match=p.steps==expected[index]['steps'] and p.failure==expected[index]['failure'] and p.colors.tolist()==expected[index]['colors']
    sampler=None
    if cp['teacher']=='JOHANSSON':
        from dcl.localpalettes.adapter import first_state,choices
        vertices=np.flatnonzero(p.colors<0)
        rng=np.random.default_rng(20261025)
        uniforms=rng.random(8192).astype(np.float32)
        sampler=[]
        with torch.inference_mode():
            for v in vertices:
                h,a=first_state(model.core,torch.tensor(p.colors),torch.tensor(p.edges),torch.tensor([v]),int(p.palette[v]))
                cs=choices(model.core,(h[0],a[0]),uniforms)
                counts=np.bincount(cs,minlength=p.k)
                sampler.append(dict(vertex=int(v),available=np.flatnonzero(availability(p,p.snapshot())[v]).tolist(),
                    palette=int(p.palette[v]),samples=len(cs),counts=counts.tolist(),
                    max_mass=float(counts.max()/len(cs)),distinct=int(np.sum(counts>0))))
    result=dict(checkpoint=cp['id'],sha256=cp['sha256'],condition=condition,
        case=expected[index],case_edges=subset[target][1]['edges'],batch_start=start,batch_size=len(subset),
        replay_matches_original=bool(match),frames=frames,sampler=sampler,
        cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,
        interpretation='Diagnostic replay of the same final batch; never a replacement final or new independent sample.',
        script_sha256=file_hash(__file__))
    dest=ROOT/'failure-replays'/cp['id']/f"{condition['id']}.json"
    atomic_json(dest,result)
    print(cp['id'],condition['id'],'match',match,'frames',len(frames),flush=True)
    return str(dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    cps=json.loads((ROOT/'checkpoint-inventory.json').read_text())+json.loads((ROOT/'supplemental-checkpoint-inventory.json').read_text())
    plan=json.loads((ROOT/'confirmatory/design.json').read_text())
    targets=[]
    for cp in cps:
        if cp['teacher']=='BE_BASE_D3':continue
        for c in plan['conditions']:
            summary=ROOT/'confirmatory'/cp['id']/c['id']/'summary.json'
            if not summary.exists():continue
            r=json.loads(summary.read_text())
            if r['successes']==r['executions']:continue
            if not args.all and cp['teacher']!='JOHANSSON':continue
            dest=ROOT/'failure-replays'/cp['id']/f"{c['id']}.json"
            targets.append(str(dest) if dest.exists() else replay(cp,c,plan))
    atomic_json(ROOT/'failure-replays/index.json',targets)
