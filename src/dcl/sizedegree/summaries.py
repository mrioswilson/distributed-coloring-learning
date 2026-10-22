"""Descriptive summaries from archived whole-run observations."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import *
from .inference import measurements,STRUCTURAL
from dcl.robustdist.classifier import interval

def main():
    f=verify_frozen();alltests=[];cells=[];counts=[];witnesses=[]
    for stratum in f['hypothesis_families']:
        data=read(OUT/f'{stratum}.json');alltests.extend(data)
        for arch in ('normalized','deepsets','transformer'):
            rows=[r for r in data if r['architecture']==arch]
            counts.append(dict(stratum=stratum,architecture=arch,comparisons=len(rows),
                               robust=sum(r['reject'] for r in rows),old=sum(r['old_reject'] for r in rows),
                               both=sum(r['reject'] and r['old_reject'] for r in rows),
                               robust_only=sum(r['reject'] and not r['old_reject'] for r in rows),
                               old_only=sum(r['old_reject'] and not r['reject'] for r in rows)))
            for n,d in sorted(set((r['n'],r['delta']) for r in rows)):
                a=[r for r in rows if r['n']==n and r['delta']==d]
                cells.append(dict(stratum=stratum,architecture=arch,n=n,delta=d,graphs=len(set(r['graph'] for r in a)),
                                  comparisons=len(a),reject=sum(r['reject'] for r in a),old_reject=sum(r['old_reject'] for r in a)))
        for r in data:
            if 'witness' in r:
                w=r['witness'];e=w['event'];name=e.get('name',e['family']+':'+','.join(map(str,e.get('ids',[]))))
                witnesses.append({k:r[k] for k in ('graph','model','architecture','stratum','n','delta')}|
                                 dict(global_reject=r['reject'],witness_name=name,witness_family=e['family'],witness_kind=e['kind'],**w))
    pd.DataFrame(counts).to_csv(OUT/'method-counts.csv',index=False)
    pd.DataFrame(cells).to_csv(OUT/'cell-test-counts.csv',index=False)
    pd.DataFrame(witnesses).to_csv(OUT/'witnesses.csv',index=False)
    atomic_json(OUT/'witnesses.json',witnesses)
    frames=[];bank_summaries=[]
    for g in graphs():
        for label in ['Luby']+[s['label'] for s in specs()]:
            a=bank(label,g['id']);good=np.flatnonzero(a['status'][:500]==0)
            row=dict(graph=g['id'],model=label,stratum=g['stage'],n=g['n'],delta=g['delta'],rep=g['rep'],
                     attempts=len(a['status']),statuses=np.bincount(a['status'],minlength=4).tolist(),
                     success_ci95=interval(int((a['status']==0).sum()),len(a['status'])))
            if not len(good):
                row.update({name+'_mean':None for name in STRUCTURAL});row['used_colors_quantiles']=None
                bank_summaries.append(row);continue
            v=measurements(a['colors'][good],g)
            row.update({name+'_mean':float(v[name].mean()) for name in STRUCTURAL})
            row['used_colors_quantiles']=np.quantile(v['used_colors'],[0,.05,.25,.5,.75,.95,1]).tolist()
            bank_summaries.append(row)
            frames.append(pd.DataFrame(dict(graph=g['id'],model=label,stratum=g['stage'],n=g['n'],delta=g['delta'],rep=g['rep'],run=good,
                                             **{name:v[name] for name in STRUCTURAL},uses_top_color=v['uses_top_color'],used_le_delta=v['used_le_delta'])))
    perrun=pd.concat(frames,ignore_index=True)
    perrun.to_csv(OUT/'all-run-statistics.csv.gz',index=False,compression='gzip')
    pd.DataFrame(bank_summaries).to_csv(OUT/'bank-descriptive-statistics.csv',index=False)
    atomic_json(OUT/'bank-descriptive-statistics.json',bank_summaries)
    # Same-state probes and actual per-run trajectory accounting.
    probes=[];trajectory=[]
    for path in sorted((OUT/'candidate-probes').glob('*.json')):
        r=read(path)
        for p in r['probes']:
            probes.append({k:r[k] for k in ('graph','model','n','delta')}|{k:v for k,v in p.items() if k not in ('q','teacher','available')})
    pd.DataFrame(probes).to_csv(OUT/'candidate-probe-summary.csv',index=False)
    gs={g['id']:g for g in graphs()}
    for t in read(ROOT/'mechanism-queue.json')['tasks']:
        g=gs[t['graph']];label='Luby' if t['model'] is None else t['model']['label']
        with np.load(ROOT/'mechanism'/f'{t["id"]}.npz') as a:
            for i in range(8):
                c=a['candidate_counts'][:,i].sum(0);p=a['participating_counts'][:,i].sum(0);co=a['commit_counts'][:,i].sum(0)
                ref=a['same_state_teacher_expected'][:,i].sum(0)
                trajectory.append(dict(graph=g['id'],model=label,n=g['n'],delta=g['delta'],run=i,status=int(a['status'][i]),
                                       candidate_top_fraction=float(c[-1]/max(c.sum(),1)),
                                       participating_top_fraction=float(p[-1]/max(p.sum(),1)),
                                       committed_top_fraction=float(co[-1]/max(co.sum(),1)),
                                       same_state_teacher_top_fraction=float(ref[-1]/max(ref.sum(),1)),
                                       top_commit_given_participation=float(co[-1]/max(p[-1],1)),
                                       other_commit_given_participation=float(co[:-1].sum()/max(p[:-1].sum(),1)),
                                       final_top_fraction=float((a['colors'][i]==g['delta']).mean()),
                                       final_used_colors=int(len(set(a['colors'][i][a['colors'][i]>=0])))))
    pd.DataFrame(trajectory).to_csv(OUT/'trajectory-accounting.csv',index=False)
    # Numerical all-free invariance is a check of the stronger finite-type argument.
    pp=pd.DataFrame(probes);initial=pp[pp['round']==0]
    equality=[]
    for (label,d),a in initial.groupby(['model','delta']):
        qs=[]
        for gid in a['graph']:
            record=read(OUT/'candidate-probes'/f'{gid}__{label}.json')
            qs.append(next(p['q'] for p in record['probes'] if p['round']==0))
        equality.append(dict(model=label,delta=int(d),identical_initial_q_across_sizes=all(np.array_equal(qs[0],q) for q in qs),graphs=len(qs)))
    atomic_json(OUT/'initial-state-invariance.json',equality)
    event=pd.DataFrame(read(OUT/'event-differences.json'))
    target=event[(event['n']==512)&(event['delta']==100)]
    target.to_csv(OUT/'target-replication.csv',index=False)
    fc=read(OUT/'factorial-contrasts.json');interactions=[]
    for arch in ('normalized','deepsets','transformer'):
        for event_name in ('used_le_delta','uses_top_color'):
            r=[r for r in fc if r['group']==arch and r['event']==event_name and r['kind']=='interaction']
            interactions.append(dict(architecture=arch,event=event_name,contrasts=len(r),significant=sum(v['significant'] for v in r),
                                     full_corners=next(v for v in r if v['contrast']=='full-corners')))
    successes=np.bincount(np.concatenate([bank(label,g['id'])['status'] for g in graphs() for label in ['Luby']+[s['label'] for s in specs()]]),minlength=4).tolist()
    result=dict(created=now(),counts=counts,interactions=interactions,statuses_confirmation=successes,
                planned_confirmation_runs=len(graphs())*16*f['total_runs_per_bank'],
                significant_holdout_witnesses=sum(w['reject'] for w in witnesses),
                holdout_witnesses_of_primary_rejections=sum(w['reject'] and next(r['reject'] for r in alltests if r['graph']==w['graph'] and r['model']==w['model']) for w in witnesses),
                positive_event_tv_bounds=int((event['tv_lower']>0).sum()),
                initial_invariance_checks=len(equality),initial_invariance_all=all(r['identical_initial_q_across_sizes'] for r in equality))
    if (OUT/'candidate-intervention/results.json').exists():
        iv=read(OUT/'candidate-intervention/results.json');ir=iv.get('rows',[])
        result['candidate_intervention']=dict(status=iv['status'],cells=len(ir),
                    approximately_equivalent=sum(r['approximate_equivalence'] for r in ir),
                    positive_effect_lower_bounds=sum(r['intervention_tv_effect_lower']>0 for r in ir))
    atomic_json(OUT/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('interactions','counts')}),flush=True)

if __name__=='__main__':main()
