"""Frozen conditional whole-coloring contrasts and separate event inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import time
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, fisher_exact
from .common import *
from .inference import *
from dcl.robustdist.stats import joint_test,multiscale_test,distance,holm,multinomial_test
from dcl.robustdist.confirmation import select_event,event_values,pending_refinements
from dcl.globaldist.core import mmd_from_kernel

def comparison(job):
    gid,label=job;f=verify_frozen();path=OUT/'tests'/f'{gid}__{label}.json'
    if path.exists():return read(path)
    g=next(g for g in graphs() if g['id']==gid);s=next(s for s in specs() if s['label']==label)
    a,b=bank('Luby',gid),bank(label,gid);m=f['test_runs_per_generator'];B=f['permutations']
    X=a['colors'][:m][a['status'][:m]==0];Y=b['colors'][:m][b['status'][:m]==0]
    out=dict(graph=gid,model=label,architecture=s['treatment'],checkpoint_seed=s['seed'],
             n=g['n'],delta=g['delta'],rho=g['rho'],family=g['family'],stratum=g['stage'],rep=g['rep'],
             checkpoint_sha256=s['sha256'],graph_sha256=g['sha256'],m_planned=m,m_Luby=len(X),m_model=len(Y),
             status_Luby=np.bincount(a['status'][:m],minlength=4).tolist(),
             status_model=np.bincount(b['status'][:m],minlength=4).tolist())
    if min(len(X),len(Y))<8:
        out.update(p=1.,old_p=1.,error='Too few valid completions for the frozen conditional test')
    else:
        q=joint_test(X,Y,g,rng('confirmation',gid,label,'omnibus'),B=B,futility=20)
        old=multiscale_test(X,Y,rng('confirmation',gid,label,'old'),B=B,futility=20,scales=(1.,))
        desc=q.pop('selected')
        out.update(p=q['p'],old_p=old['p_values']['old_hamming'],robust=q,old=old,
                   p_lower_bound=21/(B+1) if q['p_kind']=='conservative_futility' else q['p'],
                   old_p_lower_bound=21/(B+1) if old['p_kind']=='conservative_futility' else old['p_values']['old_hamming'],
                   old_mmd2_unbiased=old['statistics']['old_hamming'])
        # No new event selection uses the reserved final bank.
        event=select_event(desc,X,Y,g)
        A=a['colors'][m:][a['status'][m:]==0];C=b['colors'][m:][b['status'][m:]==0]
        vx=event_values(event,A,g);vy=event_values(event,C,g)
        ex=event_difference(int(vx.sum()),len(vx),int(vy.sum()),len(vy),.05/(2*f['total_comparisons']))
        out['witness']=dict(event=event,n_Luby=len(A),n_model=len(C),count_Luby=int(vx.sum()),count_model=int(vy.sum()),**ex)
        z=np.r_[X,Y];K=np.exp(-distance(z,z));w=np.r_[np.full(len(X),1/len(X)),np.full(len(Y),-1/len(Y))]
        mm=float(np.sqrt(max(0.,w@K@w)))
        radius=(1/np.sqrt(len(X))+1/np.sqrt(len(Y)))*(1+np.sqrt(2*np.log(40)))
        out.update(empirical_biased_mmd=mm,population_mmd_ci95=[max(0.,mm-radius),min(np.sqrt(2),mm+radius)])
        sx,sy=measurements(X,g),measurements(Y,g)
        structure=[]
        for name in STRUCTURAL:
            scale=g['delta']+1 if name=='used_colors' else (g['delta'] if name=='maximum_color' else 1.)
            wx,wy=sx[name]/scale,sy[name]/scale
            value=float(wasserstein_distance(wx,wy))
            # DKW + triangle inequality for W1 on [0,1]. Pointwise, not simultaneous.
            rad=np.sqrt(np.log(80)/(2*len(X)))+np.sqrt(np.log(80)/(2*len(Y)))
            structure.append(dict(metric=name,mean_Luby=float(sx[name].mean()),mean_model=float(sy[name].mean()),
                                  difference=float(sy[name].mean()-sx[name].mean()),normalized_wasserstein=value,
                                  wasserstein_ci95=[max(0.,value-rad),min(1.,value+rad)]))
        out['structural']=structure
    sa,sb=a['status'][:m],b['status'][:m]
    out['completion_p']=float(fisher_exact([[(sa==0).sum(),(sa!=0).sum()],[(sb==0).sum(),(sb!=0).sum()]]).pvalue)
    if sa.any() or sb.any():
        out['status_test']=multinomial_test(sa[:,None],sb[:,None],rng('status',gid,label),B=B)
        z=np.r_[a['colors'][:m],b['colors'][:m]];st=np.r_[sa,sb];ok=st==0
        K=np.exp(-distance(z,z))*(ok[:,None]&ok[None,:])
        for code in (1,2,3):K+=(st[:,None]==code)&(st[None,:]==code)
        out['coarsened_unconditional']=mmd_from_kernel(K,m,m,rng('coarsened',gid,label),B=B)
    atomic_json(path,out);return out

def refine(job):
    r,side=job;gid,label=r['graph'],r['model'];f=verify_frozen();m=f['test_runs_per_generator']
    g=next(g for g in graphs() if g['id']==gid);a,b=bank('Luby',gid),bank(label,gid)
    X=a['colors'][:m][a['status'][:m]==0];Y=b['colors'][:m][b['status'][:m]==0]
    if side=='robust':
        q=joint_test(X,Y,g,rng('confirmation',gid,label,'omnibus'),B=f['permutations']);q.pop('selected')
        r.update(robust=q,p=q['p'],p_lower_bound=q['p'])
    else:
        q=multiscale_test(X,Y,rng('confirmation',gid,label,'old'),B=f['permutations'],scales=(1.,));p=q['p_values']['old_hamming']
        r.update(old=q,old_p=p,old_p_lower_bound=p)
    r[side+'_refined_full_B']=True;atomic_json(OUT/'tests'/f'{gid}__{label}.json',r)
    return r

def elementary_events():
    f=verify_frozen();m=f['test_runs_per_generator'];rows=[];colors=[]
    cp_alpha=.025/f['elementary_event_probabilities']
    for g in graphs():
        for label in ['Luby']+[s['label'] for s in specs()]:
            if not (ROOT/'banks'/f'confirmation__{label}__{g["id"]}.npz').exists():continue
            a=bank(label,g['id']);X=a['colors'][:m][a['status'][:m]==0]
            if not len(X):
                for event in EVENTS:
                    rows.append(dict(graph=g['id'],model=label,n=g['n'],delta=g['delta'],stratum=g['stage'],
                                     event=event,count=0,total=0,probability=None,ci=[0.,1.]))
                continue
            v=measurements(X,g)
            for event in EVENTS:
                k=int(v[event].sum());n=len(X)
                rows.append(dict(graph=g['id'],model=label,n=g['n'],delta=g['delta'],stratum=g['stage'],
                                 event=event,count=k,total=n,probability=k/n,ci=interval(k,n,cp_alpha)))
            # Every color is measured prospectively. These intervals form a
            # separate simultaneous family, not post-selection naive intervals.
            for c in range(g['delta']+1):
                k=int((v['color_counts'][:,c]>0).sum());n=len(X)
                colors.append(dict(graph=g['id'],model=label,color=c,n=g['n'],delta=g['delta'],stratum=g['stage'],
                                   count=k,total=n,appearance_probability=k/n,
                                   appearance_ci=interval(k,n,.05/f['color_probabilities']),
                                   mean_vertex_fraction=float(v['color_counts'][:,c].mean()/g['n'])))
    atomic_json(OUT/'elementary-events.json',rows);pd.DataFrame(rows).to_csv(OUT/'elementary-events.csv',index=False)
    pd.DataFrame(colors).to_csv(OUT/'color-usage.csv',index=False)
    return rows

def factorial(events):
    f=verify_frozen();cs=contrast_specs(NS,DS);rows=[]
    groups={s['label']:[s['label']] for s in specs()}
    for arch in ('normalized','deepsets','transformer'):
        groups[arch]=[s['label'] for s in specs() if s['treatment']==arch]
    for event in EVENTS:
        data={(r['graph'],r['model']):r for r in events if r['event']==event and r['stratum']=='main'}
        for group,labels in groups.items():
            for kind,name,coefficients in cs:
                w=cell_weights(graphs(),labels,coefficients)
                if not all(k in data for k in w):continue
                ci=linear_interval(data,w,.025/f['factorial_contrasts'],.025/f['elementary_event_probabilities'])
                # Deterministic leave-one-graph/checkpoint sensitivity. No claim
                # that three graphs or five trainings estimate a population CI.
                sensitivity=[]
                for omit in range(f['graphs_per_cell']) if ci['estimate'] is not None else []:
                    gs=[g for g in graphs() if g['stage']!='main' or g['rep']!=omit]
                    ww=cell_weights(gs,labels,coefficients)
                    sensitivity.append(sum(v*data[k]['probability'] for k,v in ww.items()))
                for omit in labels if len(labels)>1 and ci['estimate'] is not None else []:
                    ww=cell_weights(graphs(),[x for x in labels if x!=omit],coefficients)
                    sensitivity.append(sum(v*data[k]['probability'] for k,v in ww.items()))
                rows.append(dict(event=event,group=group,kind=kind,contrast=name,
                                 coefficients={f'{n},{d}':v for (n,d),v in coefficients.items()},
                                 estimand='equal-weight fixed graph/checkpoint cohort risk-difference contrast',
                                 sensitivity_range=[min(sensitivity),max(sensitivity)] if sensitivity else None,**ci))
    atomic_json(OUT/'factorial-contrasts.json',rows)
    pd.DataFrame(rows).to_csv(OUT/'factorial-contrasts.csv',index=False)

def aggregate(workers=4):
    f=verify_frozen();allrows=[read(p) for p in sorted((OUT/'tests').glob('*.json'))]
    for stratum,size in f['hypothesis_families'].items():
        rows=[r for r in allrows if r['stratum']==stratum]
        if len(rows)!=size:raise RuntimeError(f'Incomplete {stratum}: {len(rows)}/{size}')
        for side,pkey,lower in [('robust','p','p_lower_bound'),('old','old_p','old_p_lower_bound')]:
            ids=pending_refinements(rows,pkey,lower)
            if ids:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    rr=list(pool.map(refine,[(rows[i],side) for i in ids]))
                for i,r in zip(ids,rr):rows[i]=r
        for pkey,adjusted,reject in [('p','p_holm','reject'),('old_p','old_p_holm','old_reject'),('completion_p','completion_p_holm','completion_reject')]:
            adj=holm([r[pkey] for r in rows])
            for r,p in zip(rows,adj):r[adjusted]=float(p);r[reject]=bool(p<=.05)
        witness_adj=holm([r.get('witness',{}).get('p',1.) for r in rows])
        for r,p in zip(rows,witness_adj):
            if 'witness' in r:r['witness'].update(p_holm=float(p),reject=bool(p<=.05))
        atomic_json(OUT/f'{stratum}.json',rows)
        pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]).to_csv(OUT/f'{stratum}.csv',index=False)
    events=elementary_events();lookup={(r['graph'],r['model'],r['event']):r for r in events}
    differences=[]
    for r in allrows:
        for event in EVENTS:
            a=lookup[(r['graph'],'Luby',event)];b=lookup[(r['graph'],r['model'],event)]
            q=event_difference(a['count'],a['total'],b['count'],b['total'],.025/f['elementary_event_probabilities'])
            differences.append({k:r[k] for k in ('graph','model','architecture','stratum','n','delta','rep')}|dict(event=event,count_Luby=a['count'],count_model=b['count'],m_Luby=a['total'],m_model=b['total'],**q))
    for stratum in f['hypothesis_families']:
        rows=[r for r in differences if r['stratum']==stratum];adj=holm([r['p'] for r in rows])
        for r,p in zip(rows,adj):r.update(p_holm=float(p),reject=bool(p<=.05))
    atomic_json(OUT/'event-differences.json',differences);pd.DataFrame(differences).to_csv(OUT/'event-differences.csv',index=False)
    factorial(events)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--watch',action='store_true');args=p.parse_args()
    verify_frozen()
    while True:
        jobs=[(g['id'],s['label']) for g in graphs() for s in specs()
              if (ROOT/'banks'/f'confirmation__Luby__{g["id"]}.npz').exists()
              and (ROOT/'banks'/f'confirmation__{s["label"]}__{g["id"]}.npz').exists()
              and not (OUT/'tests'/f'{g["id"]}__{s["label"]}.json').exists()]
        if jobs:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for r in pool.map(comparison,jobs):
                    print(json.dumps(dict(graph=r['graph'],model=r['model'],done=True)),flush=True)
        complete=len(list((OUT/'tests').glob('*.json')))==verify_frozen()['total_comparisons']
        if complete:
            aggregate(args.workers);print('All frozen comparisons and factorial contrasts complete',flush=True);break
        if not args.watch:break
        # Coordinator sleeps only between scans. No GPU is used here.
        time.sleep(20)

if __name__=='__main__':main()
