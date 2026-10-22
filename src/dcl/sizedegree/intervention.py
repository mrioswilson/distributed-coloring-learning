"""Separate exploratory teacher-proposal intervention with a valid coupling.

Only the candidate channel is replaced. Learned participation and COMMIT use
their original weights and numerical code. No safety observer changes actions.
Original confirmatory banks/tests are never touched by this experiment.
"""
import argparse
import signal
import time
import subprocess
from datetime import datetime
import numpy as np
import torch
from .common import *
from .run import BudgetExceeded,alarm
from dcl.globaldist.core import seed as run_seed
from dcl.localpalettes.adapter import LocalPaletteAdapter,converted_uniform
from dcl.localpalettes.runner import configure,load,persistent_gpu_lock
from dcl.universal.rules import teacher_offer,exact_commit_labels
from dcl.robustdist.classifier import interval

IROOT=ROOT/'candidate-intervention'
IOUT=OUT/'candidate-intervention'

def baseline_wall_upper():
    """Charge whole process spans, with conservative startup/teardown margins."""
    spans=[]
    for phase in ('pilot','cost'):
        q=read(ROOT/f'queue-{phase}.json')
        start=datetime.fromisoformat(q['created']);end=max(datetime.fromisoformat(t['completed']) for t in q['tasks'])
        spans.append(dict(phase=phase,seconds=(end-start).total_seconds()+60))
    q=read(ROOT/'queue-confirmation.json');end=read(ROOT/'gpu-workflow.json')['updated']
    spans.append(dict(phase='original_confirmation_and_mechanism',seconds=(datetime.fromisoformat(end)-datetime.fromisoformat(q['created'])).total_seconds()+60))
    return spans

def target_graphs():
    return [g for g in graphs() if g['family']=='regular' and g['n']==512 and g['delta']==100]

def coupled(g,model,label,stage,m):
    n=g['n'];e=np.asarray(g['edges'],np.int64);pal=np.array(g['degrees'])+1
    seeds=np.array([run_seed(stage,g['id'],label,'run',i) for i in range(m)],np.uint64)
    L=np.full((m,n),-1,np.int16);M=L.copy();sl=np.full(m,3,np.int8);sm=sl.copy()
    rl=np.zeros(m,np.int16);rm=rl.copy();unsafe=np.zeros(m,np.int64);errors=[]
    for lo in range(0,m,max(1,8192//n)):
        hi=min(m,lo+max(1,8192//n));b=hi-lo
        rngs=[np.random.default_rng(int(v)) for v in seeds[lo:hi]]
        ee=np.concatenate([e+i*n for i in range(b)],axis=1);pp=np.tile(pal,b)
        cl=np.full((b,n),-1,np.int64);cm=cl.copy();technical=False
        if model is not None:
            device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
            et=torch.as_tensor(ee,device=device);pt=torch.as_tensor(pp,device=device);adapter=LocalPaletteAdapter(model)
        try:
            for r in range(128):
                al=(cl<0).any(1);am=(cm<0).any(1)
                if not (al|am).any():break
                rl[lo:hi]+=al;rm[lo:hi]+=am
                random=np.stack([rr.random((n,2)) for rr in rngs]);rp=random[:,:,0].ravel();rc=random[:,:,1].ravel()
                fl=cl.ravel();fm=cm.ravel()
                ql,_=teacher_offer(fl,rc,ee,pp);pl=(rp<.5)&(fl<0);kl=exact_commit_labels(fl,ee,pl,ql)
                # This is the sole intervention: exact Luby proposals on the
                # intervened policy's own current state, with original rc64.
                qm,_=teacher_offer(fm,rc,ee,pp)
                if model is None:
                    pm=(rp<.5)&(fm<0);km=exact_commit_labels(fm,ee,pm,qm)
                else:
                    colors=torch.as_tensor(fm,device=device);u=(colors<0).to(dtype)
                    with torch.inference_mode():
                        p=model.participation(torch.stack((torch.as_tensor(converted_uniform(rp),device=device),u),-1)).squeeze(-1)
                        pm_t=(p>0)&(colors<0)
                        k=adapter.commit_logits(colors,et,pt,pm_t,torch.as_tensor(qm,device=device))
                    pm=pm_t.cpu().numpy();km=(k>0).cpu().numpy()&pm&(fm<0)
                # Observer only. It never masks, repairs or overrides km.
                safe=exact_commit_labels(fm,ee,pm,qm)
                unsafe[lo:hi]+=(km&~safe).reshape(b,n).sum(1)
                fl[kl]=ql[kl];fm[km]=qm[km]
        except Exception as exc:
            technical=True;errors.append(dict(lo=lo,hi=hi,error=repr(exc)))
        for dest,status,c in [(L,sl,cl),(M,sm,cm)]:
            dest[lo:hi]=c
            for j,row in enumerate(c):
                status[lo+j]=(3 if technical else 2) if (row<0).any() else (0 if (row<pal).all() and (row>=0).all() and (row[e[0]]!=row[e[1]]).all() else 1)
    equal=(sl==sm)&((sl!=0)|(L==M).all(1))
    return dict(colors_Luby=L,colors_intervened=M,status_Luby=sl,status_intervened=sm,
                rounds_Luby=rl,rounds_intervened=rm,seeds=seeds,coarsened_mismatch=~equal,
                unsafe_commits=unsafe,errors=np.array(json.dumps(errors)))

def plan():
    verify_frozen();IOUT.mkdir(parents=True,exist_ok=True)
    if (IROOT/'banks').exists():raise RuntimeError('Intervention data already exist')
    doc=Path('docs/FACTORIAL_CANDIDATE_INTERVENTION_PLAN.md')
    record=dict(created=now(),original_protocol_commit=read(OUT/'protocol-commit.json')['commit'],
                graphs=[g['id'] for g in target_graphs()],checkpoints=[s['label'] for s in specs()],
                total_cells=60,candidate_sample_sizes=[1000,500],pilot_pairs=16,budget_total_seconds=43200.,
                confidence_failure_budget=.025,conditional_tv_tolerance=.05,
                scope='exploratory proposal-channel intervention, separate from all original primary hypotheses',
                source_sha256=file_hash(__file__),plan_sha256=file_hash(doc),
                rules_sha256=file_hash('src/dcl/universal/rules.py'))
    assert len(record['graphs'])==4
    atomic_json(IOUT/'plan.json',record);print('Exploratory intervention plan saved; commit before execution',flush=True)

def worker():
    while not (ROOT/'gpu-workflow.json').exists():time.sleep(20)
    if read(ROOT/'gpu-workflow.json')['status']!='complete':
        atomic_json(IOUT/'completion.json',dict(status='skipped_original_sampling_incomplete'));return
    p=read(IOUT/'plan.json');assert file_hash(__file__)==p['source_sha256']
    assert file_hash('docs/FACTORIAL_CANDIDATE_INTERVENTION_PLAN.md')==p['plan_sha256']
    assert file_hash('src/dcl/universal/rules.py')==p['rules_sha256'];verify_frozen()
    commit=read(IOUT/'protocol-commit.json')['commit']
    assert subprocess.check_output(['git','show',commit+':src/dcl/sizedegree/intervention.py'])==Path(__file__).read_bytes()
    IROOT.mkdir(parents=True,exist_ok=True);ledger=ROOT/'budget.json';queue=IROOT/'queue.json'
    with persistent_gpu_lock():
        budget=read(ledger)
        if budget.get('active_wall'):
            budget['seconds']+=max(0.,time.time()-budget['active_wall']);budget['active_wall']=None
        spans=baseline_wall_upper();external=sum(r['seconds'] for r in spans)
        topup=max(0.,external-budget['seconds']);budget['seconds']+=topup
        budget['external_process_span_check']=dict(spans=spans,upper_seconds=external,additional_charge=topup)
        atomic_json(ledger,budget)
        if budget['limit']-budget['seconds']<180:
            atomic_json(IOUT/'completion.json',dict(status='skipped_budget',budget=budget));return
        start=time.monotonic();baseline_charge=budget['seconds'];configure();budget['seconds']+=time.monotonic()-start
        budget.setdefault('initialization_charges',[]).append(dict(phase='candidate_intervention',seconds=time.monotonic()-start))
        def finish(status,**details):
            upper=baseline_charge+time.monotonic()-start+60
            charge=max(0.,upper-budget['seconds']);budget['seconds']+=charge
            budget['intervention_process_span_check']=dict(additional_charge=charge,conservative_upper_seconds=upper)
            budget['active_wall']=None;atomic_json(ledger,budget)
            atomic_json(IOUT/'completion.json',dict(status=status,budget=budget,updated=now(),**details))
        state=read(queue) if queue.exists() else dict(tasks=[],created=now())
        pilot=state.get('pilot_tasks')
        if pilot is None:
            g=next(g for g in target_graphs() if g['stage']=='main' and g['rep']==0)
            pilot=[dict(graph=g['id'],model=s,m=16,phase='pilot',id='pilot__'+s['label']+'__'+g['id'],status='pending') for s in specs() if s['seed']==41000]
            state['pilot_tasks']=pilot
        loaded=None;model=None
        def execute(t):
            nonlocal loaded,model
            if t['status']=='complete':return True
            receipt=IROOT/'receipts'/f'{t["id"]}.json'
            if receipt.exists():
                t.update(read(receipt));return True
            left=budget['limit']-budget['seconds']
            if left<180:return False
            began=time.monotonic();budget['active_wall']=time.time();atomic_json(ledger,budget);atomic_json(queue,state)
            signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(left-150)))
            try:
                label=t['model']['label']
                if loaded!=label:
                    if model is not None:del model;torch.cuda.empty_cache()
                    model=load(t['model']);loaded=label
                g=next(g for g in target_graphs() if g['id']==t['graph'])
                a=coupled(g,model,label,'factorial-v1/candidate-intervention-'+t['phase'],t['m'])
                path=IROOT/'banks'/f'{t["id"]}.npz';save_bank(path,**a)
                t.update(status='complete',seconds=time.monotonic()-began,file=str(path),sha256=file_hash(path),completed=now())
                atomic_json(receipt,t)
                print(json.dumps(dict(id=t['id'],pairs=t['m'],seconds=t['seconds'])),flush=True);return True
            except BudgetExceeded:
                t['status']='budget_interrupted';return False
            finally:
                signal.alarm(0);budget['seconds']+=time.monotonic()-began;budget['active_wall']=None
                atomic_json(ledger,budget);atomic_json(queue,state)
        for t in pilot:
            if not execute(t):
                finish('incomplete_budget_pilot');return
        if not state['tasks']:
            cost=sum(t['seconds']/t['m'] for t in pilot)*20
            chosen=next((m for m in p['candidate_sample_sizes'] if 1.25*cost*m+180<=budget['limit']-budget['seconds']),None)
            atomic_json(IOUT/'resource-freeze.json',dict(created=now(),pairs_per_cell=chosen,
                        predicted_seconds_per_pair_all_cells=cost,remaining_seconds=budget['limit']-budget['seconds'],
                        selection='timings only; no mismatch/status/event results used'))
            if chosen is None:
                finish('skipped_budget_after_pilot');return
            state['tasks']=[dict(graph=g['id'],model=s,m=chosen,phase='confirm',
                                id='confirm__'+s['label']+'__'+g['id'],status='pending') for s in specs() for g in target_graphs()]
            atomic_json(queue,state)
        for t in state['tasks']:
            if not execute(t):break
        complete=all(t['status']=='complete' for t in state['tasks'])
        finish('complete' if complete else 'incomplete_budget',cells_completed=sum(t['status']=='complete' for t in state['tasks']),cells_planned=60)

def analyze():
    p=read(IOUT/'plan.json');completion=read(IOUT/'completion.json');rows=[]
    if not (IROOT/'queue.json').exists():
        atomic_json(IOUT/'results.json',dict(status=completion['status'],rows=[],interpretation='No intervention confirmation pairs were obtained'));return
    original={(r['graph'],r['model'],r['event']):r for r in read(OUT/'event-differences.json')}
    queue=read(IROOT/'queue.json');alpha=p['confidence_failure_budget']/(5*p['total_cells'])
    for t in queue['tasks']:
        if t['status']!='complete':continue
        assert file_hash(t['file'])==t['sha256']
        with np.load(t['file']) as a:
            m=len(a['seeds']);k=int(a['coarsened_mismatch'].sum());kl=int((a['status_Luby']==0).sum());km=int((a['status_intervened']==0).sum())
            ci=interval(k,m,alpha);cl=interval(kl,m,alpha);cm=interval(km,m,alpha)
            upper=min(1.,2*ci[1]/max(min(cl[0],cm[0]),1e-300))
            label=t['model']['label']
            lower=max(original[(t['graph'],label,e)]['tv_lower'] for e in ('used_le_delta','uses_top_color'))
            z=a['colors_intervened'][a['status_intervened']==0]
            observable={'used_le_delta':np.array([len(set(x))<=100 for x in z]),'uses_top_color':(z==100).any(1)}
            direct={};direct_lower=0.
            for event,value in observable.items():
                count=int(value.sum());cc=interval(count,len(value),alpha) if len(value) else [0.,1.]
                old=original[(t['graph'],label,event)];oc=old['ci_model']
                diff=[oc[0]-cc[1],oc[1]-cc[0]];direct_lower=max(direct_lower,0.,diff[0],-diff[1])
                direct[event]=dict(count=count,total=len(value),probability=float(value.mean()) if len(value) else None,
                                   ci=cc,original_probability=old['probability_model'],original_minus_intervention_ci=diff)
            rows.append(dict(graph=t['graph'],model=t['model']['label'],architecture=t['model']['treatment'],pairs=m,
                             mismatch_count=k,mismatch_upper=ci[1],success_count_Luby=kl,success_count_intervened=km,
                             success_lower_Luby=cl[0],success_lower_intervened=cm[0],conditional_tv_upper=upper,
                             approximate_equivalence=bool(upper<=.05),original_event_tv_lower=lower,
                             intervention_tv_effect_lower=max(0.,lower-upper,direct_lower),
                             direct_event_effect_lower=direct_lower,events=direct,unsafe_commits=int(a['unsafe_commits'].sum())))
    import pandas as pd
    atomic_json(IOUT/'results.json',dict(status=completion['status'],rows=rows,confidence=.975,
                cells_planned=60,cells_completed=len(rows),
                missing_cells=[t['id'] for t in queue['tasks'] if t['status']!='complete'],
                original_plus_intervention_joint_confidence=.95,
                estimand='Conditional complete law by R128 under an exploratory replacement of the candidate channel only',
                coupling_bound='TV(coarsened laws)<=P(coupled outputs differ); conditional TV<=2*p_diff/min(p_success)',
                causal_effect_bound='TV(original model, intervened model)>=original event-TV lower - intervened-vs-Luby TV upper'))
    pd.DataFrame(rows).to_csv(IOUT/'results.csv',index=False)
    print(json.dumps(dict(cells=len(rows),approximately_equivalent=sum(r['approximate_equivalence'] for r in rows),
                         positive_intervention_effect_bounds=sum(r['intervention_tv_effect_lower']>0 for r in rows))),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['plan','worker','analyze']);args=parser.parse_args()
    try:
        {'plan':plan,'worker':worker,'analyze':analyze}[args.phase]()
    except RuntimeError as exc:
        if args.phase=='worker' and 'Another GPU study is active' in str(exc):
            atomic_json(IOUT/'completion.json',dict(status='skipped_resource_lock',reason=str(exc),updated=now()))
        else:raise

if __name__=='__main__':main()
