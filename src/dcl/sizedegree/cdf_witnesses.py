"""Exploratory post-selection-safe CDF witnesses for preregistered metrics.

The original omnibus and original held-out event selector remain unchanged.
All seven scalar functions were frozen before sampling. Uniform bands cover
every threshold, allowing exploratory threshold/metric selection on all625
original attempts without a naive selected-event interval.
"""
import time
import numpy as np
import pandas as pd
from .common import *
from .inference import measurements,STRUCTURAL

def radius(n,alpha):
    return min(1.,float(np.sqrt(np.log(2/alpha)/(2*n)))) if n else 1.

def witness(x,y,alpha):
    x=np.sort(np.asarray(x,float));y=np.sort(np.asarray(y,float))
    if not len(x) or not len(y):
        return dict(estimable=False,threshold=None,ci=[-1.,1.],tv_lower=0.,ks=None)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    values=np.unique(np.r_[x,y]);a=np.searchsorted(x,values,side='right');b=np.searchsorted(y,values,side='right')
    p=a/len(x);q=b/len(y);diff=q-p;j=int(np.argmax(abs(diff)))
    ex,ey=radius(len(x),alpha),radius(len(y),alpha)
    cx=[max(0.,float(p[j])-ex),min(1.,float(p[j])+ex)]
    cy=[max(0.,float(q[j])-ey),min(1.,float(q[j])+ey)]
    ci=[cy[0]-cx[1],cy[1]-cx[0]]
    return dict(estimable=True,threshold=float(values[j]),m_Luby=len(x),m_model=len(y),
                count_Luby=int(a[j]),count_model=int(b[j]),probability_Luby=float(p[j]),probability_model=float(q[j]),
                difference=float(diff[j]),ci=ci,tv_lower=max(0.,ci[0],-ci[1]),ks=float(abs(diff[j])),
                probability_ci_Luby=cx,probability_ci_model=cy,epsilon_Luby=ex,epsilon_model=ey)

def main():
    began=time.monotonic();f=verify_frozen();gs=graphs();labels=['Luby']+[s['label'] for s in specs()]
    assert len(labels)==len(set(labels))==16 and len(STRUCTURAL)==7
    family=len(gs)*len(labels)*len(STRUCTURAL);alpha=.05/family
    tests={}
    for stage in f['hypothesis_families']:
        tests.update({(r['graph'],r['model']):r for r in read(OUT/f'{stage}.json')})
    rows=[];best=[];cdf_count=0
    for g in gs:
        a=bank('Luby',g['id']);assert len(a['status'])==f['total_runs_per_bank']
        X=a['colors'][a['status']==0];vx=measurements(X,g) if len(X) else None;cdf_count+=len(STRUCTURAL)
        for label in labels[1:]:
            b=bank(label,g['id']);assert len(b['status'])==f['total_runs_per_bank']
            Y=b['colors'][b['status']==0];vy=measurements(Y,g) if len(Y) else None;cdf_count+=len(STRUCTURAL)
            r=tests[(g['id'],label)];group=[]
            for metric in STRUCTURAL:
                w=witness(vx[metric] if vx else [],vy[metric] if vy else [],alpha)
                q=dict(graph=g['id'],model=label,architecture=r['architecture'],stratum=g['stage'],n=g['n'],delta=g['delta'],
                       metric=metric,event='metric(X) <= threshold',global_reject=r['reject'],**w)
                rows.append(q);group.append(q)
            best.append(max(group,key=lambda r:(r['tv_lower'],r['ks'] if r['ks'] is not None else -1)))
    assert cdf_count==family and len(best)==f['total_comparisons'] and len(rows)==len(best)*7
    counts=[]
    for stage in f['hypothesis_families']:
        for arch in ('normalized','deepsets','transformer'):
            rr=[r for r in best if r['stratum']==stage and r['architecture']==arch]
            counts.append(dict(stratum=stage,architecture=arch,cells=len(rr),positive_tv_bounds=sum(r['tv_lower']>0 for r in rr),
                               positive_for_global_rejections=sum(r['tv_lower']>0 and r['global_reject'] for r in rr)))
    record=dict(analysis_type='Exploratory, introduced after inspecting original outcomes and explanation failures',
                simultaneous_confidence=.95,cdf_family=family,alpha_per_CDF=alpha,metrics=list(STRUCTURAL),
                scalar_function_source_sha256=f['source_hashes']['src/dcl/sizedegree/inference.py'],
                sample_scope='All625 original attempts per bank; conditional proper completions by R128',
                source_sha256=file_hash(__file__),counts=counts,seconds=time.monotonic()-began,
                warning='Separate95% family; no joint95% guarantee with original tests, CP witnesses or intervention families')
    atomic_json(OUT/'exploratory-cdf-protocol-and-summary.json',record)
    atomic_json(OUT/'exploratory-cdf-witnesses.json',rows);pd.DataFrame(rows).to_csv(OUT/'exploratory-cdf-witnesses.csv',index=False)
    atomic_json(OUT/'exploratory-cdf-best-witnesses.json',best)
    # Record the observed resubstitution failure without changing that selector.
    gid='main-regular-n1024-d50-r0';label='normalized-s41003';g=next(g for g in gs if g['id']==gid)
    event=tests[(gid,label)]['witness']['event']
    from dcl.robustdist.confirmation import event_values
    diagnostic=dict(graph=gid,model=label,event_kind=event['kind'],event_family=event['family'],ids=event.get('ids'),
                    observed_keys=len(event.get('keys',[])),positive_keys=sum(v>0 for v in event.get('scores',[])))
    for who in ('Luby',label):
        b=bank(who,gid);Z=b['colors'][:500][b['status'][:500]==0];H=b['colors'][500:][b['status'][500:]==0]
        v=event_values(event,Z,g);h=event_values(event,H,g);cut=len(Z)//2
        diagnostic[who]=dict(selection_count=int(v[:cut].sum()),selection_n=cut,
                             evaluation_count=int(v[cut:].sum()),evaluation_n=len(Z)-cut,
                             reserved_count=int(h.sum()),reserved_n=len(H))
    atomic_json(OUT/'explanation-selection-diagnostic.json',diagnostic)
    print(json.dumps(dict(cdf_family=family,comparisons=len(best),counts=counts,seconds=time.monotonic()-began)),flush=True)

if __name__=='__main__':main()
