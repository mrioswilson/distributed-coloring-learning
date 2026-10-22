from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,binom,hypergeom
from .core import *
from dcl.study.storage import atomic_json

def read(p):return json.loads(Path(p).read_text())
def bank(stage,label,g):return np.load(ROOT/'banks'/f'{stage}__{label}__{g}.npz',allow_pickle=False)
def successes(a):return a['colors'][a['status']==0]
def wilson(k,n,confidence=.95):
    if n==0:return [0.,1.]
    z=norm.ppf(1-(1-confidence)/2);p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,mid-radius)),float(min(1,mid+radius))]

def structural(x,g):
    if not len(x):return dict(error='no successes')
    k=max(g['degrees'])+1
    counts=np.stack([np.bincount(y,minlength=k) for y in x]);used=(counts>0).sum(1)
    largest=counts.max(1)/g['n'];entropy=-(np.where(counts>0,counts/g['n']*np.log(np.maximum(counts,1)/g['n']),0)).sum(1)
    e=set(map(tuple,np.array(g['edges']).T.tolist()));pairs=[]
    for u in range(g['n']):
        for v in range(u+1,g['n']):
            if (u,v) not in e:pairs.append((u,v))
    rng=np.random.default_rng(seed('diagnostic-pairs',g['id']))
    if len(pairs)>256:pairs=[pairs[i] for i in rng.choice(len(pairs),256,replace=False)]
    coc=np.array([np.mean(x[:,u]==x[:,v]) for u,v in pairs])
    partitions=[]
    for y in x:
        mapping={};p=[]
        for c in y:
            if int(c) not in mapping:mapping[int(c)]=len(mapping)
            p.append(mapping[int(c)])
        partitions.append(tuple(p))
    perdegree={}
    for d in sorted(set(g['degrees'])):
        ids=np.flatnonzero(np.array(g['degrees'])==d)
        # Each row first aggregates vertices of this fixed degree; uncertainty is across runs.
        h=np.stack([np.bincount(y[ids],minlength=k)/len(ids) for y in x])
        perdegree[str(d)]=dict(mean=h.mean(0).tolist(),sd_across_runs=h.std(0).tolist())
    return dict(used_mean=float(used.mean()),used_sd=float(used.std()),used_values=used.tolist(),
                largest_class_mean=float(largest.mean()),class_entropy_mean=float(entropy.mean()),
                sorted_class_sizes_mean=np.sort(counts,axis=1)[:,::-1].mean(0).tolist(),
                nonadjacent_pairs=pairs,cocolor_mean=coc.tolist(),partition_unique_fraction=len(set(partitions))/len(x),
                color_by_degree=perdegree)

def controls():
    target=OUT/'controls.json'
    if target.exists():return read(target)
    rows=[]
    for g in ('path3','regular-n256-d100'):
        a=bank('negative-luby','Luby',g)
        assert not a['status'].any()
        for i in range(50):
            x=a['colors'][200*i:200*i+100];y=a['colors'][200*i+100:200*i+200]
            r=mmd_test(x,y,np.random.default_rng(seed('control-perm','Luby',g,i)),B=199)
            rows.append(dict(kind='Luby/Luby',graph=g,trial=i,p=r['p']))
    for s in specs():
        a=bank('negative-model',s['label'],'path3');assert not a['status'].any()
        for i in range(30):
            x=a['colors'][200*i:200*i+100];y=a['colors'][200*i+100:200*i+200]
            r=mmd_test(x,y,np.random.default_rng(seed('control-perm',s['label'],i)),B=199)
            rows.append(dict(kind='same checkpoint',model=s['label'],graph='path3',trial=i,p=r['p']))
    x=valid_colors(small_graphs()[1])
    for i in range(100):
        r=mmd_test(np.repeat(x[:1],100,0),np.repeat(x[-1:],100,0),np.random.default_rng(seed('positive',i)),B=199)
        rows.append(dict(kind='known distinct valid point masses',trial=i,graph='path3',p=r['p']))
    summary=[]
    for kind in sorted(set(r['kind'] for r in rows)):
        a=[r for r in rows if r['kind']==kind];k=sum(r['p']<=.05 for r in a);n=len(a)
        summary.append(dict(kind=kind,rejections=k,trials=n,rate=k/n,ci95=wilson(k,n),
            compatible_with_nominal_null=(int(binom.ppf(.005,n,.05))<=k<=int(binom.ppf(.995,n,.05))) if 'point masses' not in kind else None))
    result=dict(rows=rows,summary=summary);atomic_json(target,result);return result

def primary():
    gs=read(OUT/'graphs.json');ss=read(OUT/'checkpoints.json');rows=[];struct=[];unconditional=[]
    directory=OUT/'tests';directory.mkdir(parents=True,exist_ok=True)
    for g in gs:
        teacher=bank('confirmatory','Luby',g['id']);x=successes(teacher)
        struct.append(dict(graph=g['id'],model='Luby',**structural(x,g)))
        for s in ss:
            path=directory/f"{g['id']}__{s['label']}.json"
            a=bank('confirmatory',s['label'],g['id']);y=successes(a)
            if path.exists():r=read(path)
            else:
                ps=seed('primary-permutation',g['id'],s['label'])
                r=mmd_test(x,y,np.random.default_rng(ps),B=19999)
                if 'error' in r:r.update(p=1,m=len(x),n=len(y))
                r.update(graph=g['id'],model=s['label'],architecture=s['treatment'],training_seed=s['seed'],n_vertices=g['n'],
                         delta=g['delta'],family=g['family'],permutation_seed=ps,
                         teacher_status=np.bincount(teacher['status'],minlength=4).tolist(),
                         model_status=np.bincount(a['status'],minlength=4).tolist(),planned_m=250)
                atomic_json(path,r)
            rows.append(r);struct.append(dict(graph=g['id'],model=s['label'],**structural(y,g)))
            if a['status'].any() or teacher['status'].any():
                # Finite orthogonal category atoms plus kernel on valid vectors.
                z=np.concatenate([teacher['colors'],a['colors']]);st=np.r_[teacher['status'],a['status']]
                K=kernel(z,z);valid=st==0
                K*=valid[:,None]&valid[None,:]
                for c in (1,2,3):K+=(st[:,None]==c)&(st[None,:]==c)
                u=mmd_from_kernel(K,250,250,np.random.default_rng(seed('unconditional',g['id'],s['label'])),B=19999)
                # Pure status kernel is a squared L2 difference between category masses.
                S=(st[:,None]==st[None,:]).astype(float)
                v=mmd_from_kernel(S,250,250,np.random.default_rng(seed('statuses',g['id'],s['label'])),B=19999)
                unconditional.extend([dict(graph=g['id'],model=s['label'],kind=kind,**r0) for kind,r0 in [('coarsened-unconditional',u),('status',v)]])
        print('Primary analyzed',g['id'],flush=True)
    adj=holm([r['p'] for r in rows])
    for r,p in zip(rows,adj):
        r.update(p_holm=float(p),reject_holm=bool(p<=.05),alpha=.05,
                 equivalent_mmd_002=bool(r.get('mmd_population_ci95',[0,2])[1]<.02))
    atomic_json(OUT/'primary.json',rows)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]).to_csv(OUT/'primary.csv',index=False)
    atomic_json(OUT/'structural.json',struct)
    pd.DataFrame([{k:v for k,v in r.items() if k in ('graph','model','used_mean','used_sd','largest_class_mean','class_entropy_mean','partition_unique_fraction')} for r in struct]).to_csv(OUT/'structural.csv',index=False)
    if unconditional:
        adj=holm([r['p'] for r in unconditional])
        for r,p in zip(unconditional,adj):r['p_holm']=float(p)
    atomic_json(OUT/'unconditional.json',unconditional)
    return rows

def small():
    rows=[];laws=[]
    for g in small_graphs():
        x,p,r,states=exact_teacher(g)
        atomic_json(OUT/f"exact-{g['id']}.json",dict(graph=g['id'],states=states,rational=r,support=x.tolist(),probabilities=p.tolist(),
                                                   tv_to_uniform=float(np.abs(p-1/len(p)).sum()/2)))
        for label in ['Luby']+[s['label'] for s in specs()]:
            a=bank('estimation',label,g['id']);y=successes(a)
            counts=np.array([np.sum((y==xx).all(1)) for xx in x]);assert counts.sum()==len(y)
            tv,ci=tv_interval(counts,p)
            rows.append(dict(graph=g['id'],model=label,n=len(y),tv=tv,tv_ci_low=ci[0],tv_ci_high=ci[1],
                             tv_uniform=float(np.abs(counts/len(y)-1/len(x)).sum()/2),support_size=len(x)))
            laws.append(dict(graph=g['id'],model=label,support=x.tolist(),probabilities=(counts/len(y)).tolist(),counts=counts.tolist(),teacher_probabilities=p.tolist()))
    atomic_json(OUT/'small-laws.json',laws);pd.DataFrame(rows).to_csv(OUT/'small-tv.csv',index=False)
    return rows

def classifier():
    path=OUT/'classifier.json'
    if path.exists():return read(path)
    rows=[];gs={g['id']:g for g in read(OUT/'graphs.json')}
    for g in ('path3','star-d200','regular-n256-d100'):
        for s in [s for s in specs() if s['seed']==41000]:
            aa=bank('classifier','Luby',g);bb=bank('classifier',s['label'],g)
            if aa['status'].any() or bb['status'].any():
                rows.append(dict(graph=g,model=s['label'],error='failure in dedicated classifier banks',p=1));continue
            a=aa['colors'];b=bb['colors'];n=a.shape[1];k=max(gs[g]['degrees'])+1
            def fit(alpha):
                logp=[]
                for x in (a[:100],b[:100]):
                    h=np.stack([np.bincount(x[:,v],minlength=k) for v in range(n)])
                    logp.append(np.log((h+alpha)/(100+k*alpha)))
                return logp
            def predict(logp,x):
                scores=np.array([lp[np.arange(n)[None,:],x].sum(1) for lp in logp]);return scores[1]>scores[0]
            candidates=[]
            for alpha in (.5,1.):
                lp=fit(alpha);val=(np.mean(~predict(lp,a[100:200]))+np.mean(predict(lp,b[100:200])))/2
                candidates.append((val,alpha,lp))
            best=max(candidates,key=lambda z:(z[0],-z[1]));val,alpha,lp=best
            pa=predict(lp,a[200:300]);pb=predict(lp,b[200:300]);tn=int((~pa).sum());tp=int(pb.sum())
            acc=(tn+tp)/200;prediction_total=int(pa.sum()+pb.sum())
            rng=np.random.default_rng(seed('classifier-test-permutation',g,s['label']))
            random_tp=rng.hypergeometric(prediction_total,200-prediction_total,100,size=9999)
            perm_accuracy=(2*random_tp+100-prediction_total)/200
            p=(1+int(np.sum(perm_accuracy>=acc-1e-14)))/10000
            ca=wilson(tn,100,.975);cb=wilson(tp,100,.975)
            rows.append(dict(graph=g,model=s['label'],architecture=s['treatment'],smoothing=alpha,validation_accuracy=val,
                             balanced_accuracy=acc,ci95=[(ca[0]+cb[0])/2,(ca[1]+cb[1])/2],p=p,train_per_class=100,validation_per_class=100,test_per_class=100))
    for r,p in zip(rows,holm([r['p'] for r in rows])):r['p_holm']=float(p)
    atomic_json(path,rows);pd.DataFrame(rows).to_csv(OUT/'classifier.csv',index=False);return rows

def power():
    directory=OUT/'power';directory.mkdir(parents=True,exist_ok=True);rows=[]
    for g in ('path3','star-d200','regular-n256-d100'):
        for s in [s for s in specs() if s['seed']==41000]:
            path=directory/f"{g}__{s['label']}.json"
            if path.exists():rows.extend(read(path));continue
            a=successes(bank('power','Luby',g));b=successes(bank('power',s['label'],g))
            z=np.concatenate((a,b));K=kernel(z,z);local=[]
            for m in (100,250,500,1000,2000):
                pv=[]
                for trial in range(30):
                    rng=np.random.default_rng(seed('power-resampling',g,s['label'],m,trial))
                    idx=np.r_[rng.integers(0,len(a),m),len(a)+rng.integers(0,len(b),m)]
                    r=mmd_from_kernel(K[np.ix_(idx,idx)],m,m,rng,B=199);pv.append(r['p'])
                k=sum(p<=.05 for p in pv);ci=wilson(k,len(pv))
                local.append(dict(graph=g,model=s['label'],architecture=s['treatment'],m=m,rejections=k,trials=30,
                                  conditional_empirical_bank_power=k/30,ci_low=ci[0],ci_high=ci[1],p_values=pv,
                                  interpretation='Conditional empirical-bank resampling; CI excludes bank-law uncertainty; nominal alpha=.05, not Holm-adjusted'))
                print('Power',g,s['label'],m,k,flush=True)
            atomic_json(path,local);rows.extend(local)
    atomic_json(OUT/'power.json',rows);pd.DataFrame(rows).drop(columns='p_values').to_csv(OUT/'power.csv',index=False);return rows

def main():
    c=controls()
    if any(r['compatible_with_nominal_null'] is False for r in c['summary']):
        raise RuntimeError('Negative control calibration requires investigation; raw controls retained')
    primary();small();classifier();power()
if __name__=='__main__':main()

def independent_power():
    target=OUT/'independent-power.json'
    if target.exists():return read(target)
    rows=[]
    for s in [s for s in specs() if s['seed']==41000]:
        a=bank('independent-power','Luby','path3');b=bank('independent-power',s['label'],'path3');offset=0
        for m in (100,250,500,1000,2000):
            pv=[];sizes=[]
            for trial in range(20):
                take=slice(offset,offset+m);offset+=m
                x=a['colors'][take][a['status'][take]==0];y=b['colors'][take][b['status'][take]==0]
                sizes.append([len(x),len(y)])
                r=mmd_test(x,y,np.random.default_rng(seed('independent-power-perm',s['label'],m,trial)),B=199);pv.append(r.get('p',1))
            k=sum(p<=.05 for p in pv);ci=wilson(k,20)
            rows.append(dict(graph='path3',model=s['label'],architecture=s['treatment'],m=m,trials=20,rejections=k,
                             independent_population_power=k/20,ci_low=ci[0],ci_high=ci[1],p_values=pv,actual_success_sizes=sizes))
            print('Independent power',s['label'],m,k,flush=True)
        assert offset==77000
    atomic_json(target,rows);pd.DataFrame(rows).drop(columns=['p_values','actual_success_sizes']).to_csv(OUT/'independent-power.csv',index=False)
    return rows
