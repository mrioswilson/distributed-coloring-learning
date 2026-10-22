"""Frozen comparisons, independent explanatory events, and exact small laws."""
import argparse,time
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from .common import *
from .stats import joint_test,multiscale_test,multinomial_test,evaluate,distance,holm
from .classifier import classifier_test,interval
from dcl.globaldist.core import exact_teacher,tv_interval,mmd_from_kernel

@lru_cache(maxsize=64)
def loadbank(label,graph):
    file=DATA/'banks'/f'confirmatory__{label}__{graph}.npz'
    a=np.load(file);return {k:a[k] for k in ('colors','status','seeds')}

def event_values(desc,z,g):
    if desc['kind']=='categorical':
        keys={tuple(k) for k,s in zip(desc['keys'],desc['scores']) if s>0}
        return np.array([tuple(row) in keys for row in z[:,desc['ids']]])
    h=evaluate([desc],z,g)[:,0]
    return h.astype(bool) if desc['kind'] in ('threshold','structural_threshold') else h<=desc['event_threshold']

def select_event(descs,x,y,g):
    candidates=[]
    for desc in descs:
        if desc['kind']=='dense':continue
        desc=dict(desc)
        if desc['kind']=='structural':
            h=evaluate([desc],np.r_[x,y],g)[:,0];values=np.unique(h)
            thresholds=np.unique(np.quantile(values,np.linspace(0,1,17)))
            diffs=[abs((h[:len(x)]<=v).mean()-(h[len(x):]<=v).mean()) for v in thresholds]
            desc['event_threshold']=float(thresholds[int(np.argmax(diffs))])
        p=event_values(desc,x,g).mean();q=event_values(desc,y,g).mean()
        candidates.append((abs(p-q),desc))
    return max(candidates,key=lambda a:a[0])[1]

def explanation(label,g,descs,x,y):
    # Selection may use all primary labels. Estimation uses only the reserved bank.
    event=select_event(descs,x,y,g);a=loadbank('Luby',g['id']);b=loadbank(label,g['id'])
    X=a['colors'][-250:][a['status'][-250:]==0];Y=b['colors'][-250:][b['status'][-250:]==0]
    if min(len(X),len(Y))==0:return dict(event=event,p=1.,error='No valid reserved explanation sample',n_Luby=len(X),n_model=len(Y))
    px=event_values(event,X,g);py=event_values(event,Y,g);kx=int(px.sum());ky=int(py.sum())
    alpha=.05/(2*795);cx=interval(kx,len(X),alpha);cy=interval(ky,len(Y),alpha)
    p=float(fisher_exact([[kx,len(X)-kx],[ky,len(Y)-ky]]).pvalue)
    return dict(event=event,n_Luby=len(X),n_model=len(Y),count_Luby=kx,count_model=ky,probability_Luby=kx/len(X),probability_model=ky/len(Y),
                difference=ky/len(Y)-kx/len(X),ci_Luby=cx,ci_model=cy,difference_ci=[cy[0]-cx[1],cy[1]-cx[0]],p=p,
                selection='primary data only; estimation on final reserved 250 runs; simultaneous CP bounds over 1590 probabilities')

def comparison(job):
    gid,label,m=job;path=OUT/'confirmation-tests'/f'{gid}__{label}__m{m}.json'
    if path.exists():return read(path)
    g=next(g for g in graphs() if g['id']==gid);spec=next(s for s in specs() if s['label']==label)
    a,b=loadbank('Luby',gid),loadbank(label,gid);X=a['colors'][:m][a['status'][:m]==0];Y=b['colors'][:m][b['status'][:m]==0]
    if min(len(X),len(Y))<8:
        result=dict(graph=gid,model=label,total_per_generator=m,p=1.,old_p=1.,error='insufficient valid completions',architecture=spec['treatment'],training_seed=spec['seed'],n=g['n'],delta=g['delta'],family=g['family'])
    else:
        robust=joint_test(X,Y,g,rng('confirmatory',gid,label,m,'omnibus'),B=19999,futility=20)
        old=multiscale_test(X,Y,rng('confirmatory',gid,label,m,'old'),B=19999,futility=20,scales=(1.,))
        z=np.r_[X,Y];K=np.exp(-distance(z,z));w=np.r_[np.full(len(X),1/len(X)),np.full(len(Y),-1/len(Y))]
        norm=float(np.sqrt(max(0.,w@K@w)));radius=(1/np.sqrt(len(X))+1/np.sqrt(len(Y)))*(1+np.sqrt(2*np.log(40)))
        descs=robust.pop('selected');result=dict(graph=gid,model=label,architecture=spec['treatment'],training_seed=spec['seed'],n=g['n'],delta=g['delta'],family=g['family'],
            total_per_generator=m,p=robust['p'],old_p=old['p_values']['old_hamming'],robust=robust,old=old,
            p_lower_bound=21/20000 if robust['p_kind']=='conservative_futility' else robust['p'],
            old_p_lower_bound=21/20000 if old['p_kind']=='conservative_futility' else old['p_values']['old_hamming'],
            old_empirical_mmd=norm,old_population_mmd_ci95=[max(0.,norm-radius),min(np.sqrt(2),norm+radius)],
            graph_sha256=g['sha256'],checkpoint_sha256=spec['sha256'],
            statuses_Luby=np.bincount(a['status'][:m],minlength=4).tolist(),statuses_model=np.bincount(b['status'][:m],minlength=4).tolist())
        if m==500:result['explanation']=explanation(label,g,descs,X,Y)
        if m==500 and spec['seed']==41000 and gid in ('path3','star-d200','regular-n256-d100'):
            k=min(len(X),len(Y));result['classifier']=classifier_test(X[:k],Y[:k],g)
    sa,sb=a['status'][:m],b['status'][:m]
    if sa.any() or sb.any():
        result['status_test']=multinomial_test(sa[:,None],sb[:,None],rng('status-law',gid,label,m),B=19999)
        z=np.r_[a['colors'][:m],b['colors'][:m]];st=np.r_[sa,sb];ok=st==0
        K=np.exp(-distance(z,z))*(ok[:,None]&ok[None,:])
        for code in (1,2,3):K+=(st[:,None]==code)&(st[None,:]==code)
        result['coarsened_unconditional']=mmd_from_kernel(K,m,m,rng('unconditional-law',gid,label,m),B=19999)
    atomic_json(path,result);return result

def pending_refinements(rows,pkey,lowerkey):
    lower=[r.get(lowerkey,r[pkey]) for r in rows];adj=holm(lower)
    return [i for i,r in enumerate(rows) if adj[i]<=.05 and r[pkey]>lower[i]]

def refine_one(job):
    r,side=job;gid,label,m=r['graph'],r['model'],r['total_per_generator']
    g=next(g for g in graphs() if g['id']==gid);a,b=loadbank('Luby',gid),loadbank(label,gid)
    X=a['colors'][:m][a['status'][:m]==0];Y=b['colors'][:m][b['status'][:m]==0]
    if side=='robust':
        q=joint_test(X,Y,g,rng('confirmatory',gid,label,m,'omnibus'),B=19999)
        q.pop('selected');r.update(robust=q,p=q['p'],p_lower_bound=q['p'])
    else:
        q=multiscale_test(X,Y,rng('confirmatory',gid,label,m,'old'),B=19999,scales=(1.,))
        p=q['p_values']['old_hamming'];r.update(old=q,old_p=p,old_p_lower_bound=p)
    r[side+'_refined_full_B']=True
    atomic_json(OUT/'confirmation-tests'/f'{gid}__{label}__m{m}.json',r)
    return r

def small_reference(job):
    gid,label=job;path=OUT/'small-tests'/f'{gid}__{label}.json'
    if path.exists():return read(path)
    g=next(g for g in graphs() if g['id']==gid);support,p,law,states=exact_teacher(g)
    a,b=loadbank('Luby',gid),loadbank(label,gid);X=a['colors'][:2000][a['status'][:2000]==0];Y=b['colors'][:2000][b['status'][:2000]==0]
    k=min(len(X),len(Y));direct=multinomial_test(X[:k],Y[:k],rng('small-direct',gid,label),B=19999)
    mapping={tuple(v):i for i,v in enumerate(support)};counts=np.bincount([mapping[tuple(y)] for y in Y],minlength=len(p))
    tv,ci=tv_interval(counts,p,alpha=.05/105);tail=min(1.,g['n']*(1-2.**(-g['delta']-1))**128)
    ci=[max(0.,ci[0]-tail),min(1.,ci[1]+tail)]
    r=dict(graph=gid,model=label,architecture=label.split('-s')[0],tv_estimate_vs_exact_eventual_teacher=tv,simultaneous_tv_ci=ci,
           teacher_cutoff_tv_bound=tail,tolerance=.10,approximate_equivalence=bool(ci[1]<=.1),support_size=len(p),valid_model=len(Y),direct=direct,
           exact_law=law,reachable_states=states)
    atomic_json(path,r);return r

def aggregate():
    old={(r['graph'],r['model']):r for r in read(OLD/'results/ucgnn-global-distribution/primary.json')}
    allrows=[read(p) for p in (OUT/'confirmation-tests').glob('*.json')];primary=[r for r in allrows if r['total_per_generator']==500]
    for rows,name in [(primary,'primary'),([r for r in allrows if r['total_per_generator']!=500],'secondary')]:
        if not rows:continue
        for side,pkey,lowerkey in [('robust','p','p_lower_bound'),('old','old_p','old_p_lower_bound')]:
            ids=pending_refinements(rows,pkey,lowerkey)
            if ids:
                with ProcessPoolExecutor(max_workers=4) as pool:refined=list(pool.map(refine_one,[(rows[i],side) for i in ids]))
                for i,r in zip(ids,refined):rows[i]=r
        adj=holm([r['p'] for r in rows]);oldadj=holm([r['old_p'] for r in rows])
        for r,p,q in zip(rows,adj,oldadj):
            h=old[(r['graph'],r['model'])];r.update(p_holm=float(p),reject=bool(p<=.05),old_p_holm=float(q),old_reject=bool(q<=.05),historical_p=h['p'],historical_p_holm=h['p_holm'],historical_reject=h['reject_holm'])
        atomic_json(OUT/f'{name}.json',rows)
        pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows]).to_csv(OUT/f'{name}.csv',index=False)
    ex=[]
    for r in primary:
        if 'explanation' in r:ex.append({k:r[k] for k in ('graph','model','architecture','n','delta','family','reject')}|r['explanation'])
    if ex:
        adj=holm([r['p'] for r in ex])
        for r,p in zip(ex,adj):r['p_holm']=float(p);r['witness_reject']=bool(p<=.05)
        atomic_json(OUT/'explanatory-witnesses.json',ex);pd.DataFrame(ex).to_csv(OUT/'explanatory-witnesses.csv',index=False)
    classifiers=[dict(graph=r['graph'],model=r['model'],architecture=r['architecture'],**r['classifier']) for r in primary if 'classifier' in r]
    if classifiers:
        adj=holm([r['p'] for r in classifiers])
        for r,p in zip(classifiers,adj):r['p_holm']=float(p)
        atomic_json(OUT/'classifiers.json',classifiers);pd.DataFrame(classifiers).to_csv(OUT/'classifiers.csv',index=False)
    failures=[]
    for r in primary:
        for name in ('status_test','coarsened_unconditional'):
            if name in r:failures.append(dict(graph=r['graph'],model=r['model'],kind=name,**r[name]))
    for name in ('status_test','coarsened_unconditional'):
        rr=[r for r in failures if r['kind']==name];adj=holm([r['p'] for r in rr]+[1.]*(795-len(rr)))[:len(rr)]
        for r,p in zip(rr,adj):r['p_holm']=float(p)
    atomic_json(OUT/'failure-law-tests.json',failures)
    small=[read(p) for p in (OUT/'small-tests').glob('*.json')]
    if small:
        adj=holm([r['direct']['p'] for r in small])
        for r,p in zip(small,adj):r['p_holm']=float(p);r['reject']=bool(p<=.05)
        atomic_json(OUT/'small-reference.json',small)
        pd.DataFrame([{k:v for k,v in r.items() if k not in ('direct','exact_law')}|dict(direct_p=r['direct']['p']) for r in small]).to_csv(OUT/'small-reference.csv',index=False)
    return primary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    freeze=read(OUT/'freeze.json');ext=set(freeze['extended_graphs']);jobs=[(g['id'],s['label'],m) for g in graphs() for s in specs() for m in ([500,250,1000,2000] if g['id'] in ext else [500])]
    smalljobs=[(g['id'],s['label']) for g in graphs() if g['stage']=='small' for s in specs()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        while jobs or smalljobs:
            ready=[]
            for job in jobs:
                gid,label,m=job
                if (DATA/'receipts'/f'confirmatory__Luby__{gid}.json').exists() and (DATA/'receipts'/f'confirmatory__{label}__{gid}.json').exists():ready.append(job)
            for r in pool.map(comparison,ready):print('Compared',r['graph'],r['model'],r['total_per_generator'],flush=True)
            jobs=[j for j in jobs if j not in ready]
            smallready=[j for j in smalljobs if (DATA/'receipts'/f'confirmatory__Luby__{j[0]}.json').exists() and (DATA/'receipts'/f'confirmatory__{j[1]}__{j[0]}.json').exists()]
            for r in pool.map(small_reference,smallready):print('Small reference',r['graph'],r['model'],flush=True)
            smalljobs=[j for j in smalljobs if j not in smallready]
            if jobs or smalljobs:
                if not args.watch:raise RuntimeError('Required confirmation banks incomplete')
                time.sleep(10)
    aggregate();print('All frozen comparisons complete',flush=True)
if __name__=='__main__':main()
