"""Transfer checks; existing extensive V2 validation remains the main reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import binomtest
from dcl.robustdist.stats import joint_test,multiscale_test
from dcl.robustdist.controls_v2 import draw
from dcl.robustdist.common import rng as old_rng
from dcl.robustdist.classifier import interval
from .common import *

def trial(job):
    case,t=job
    path=OUT/'validation-trials'/f'{case}-{t:04d}.json'
    if path.exists():return read(path)
    if case=='star_positive':
        x,g,_=draw('localized',500,rng('transfer',case,t,'A'),0)
        y,_,_=draw('localized',500,rng('transfer',case,t,'B'),1)
        B=19999;alpha=.05/720
    elif case=='diffuse_positive':
        x,g,_=draw('diffuse',500,rng('transfer',case,t,'A'),0)
        y,_,_=draw('diffuse',500,rng('transfer',case,t,'B'),1)
        B=19999;alpha=.05/720
    elif case=='point_null':
        x,g,_=draw('point_null',100,rng('transfer',case,t,'A'))
        y=x.copy();B=199;alpha=.05
    else:
        label='normalized-s41000' if case=='neural_empirical_null' else 'Luby'
        n,d=(1024,100) if case=='large_empirical_null' else (128,3)
        g=next(g for g in read(OUT/'pilot-graphs.json') if g['n']==n and g['delta']==d)
        a=bank(label,g['id'],'pilot');pool=a['colors'][a['status']==0]
        x=pool[rng('transfer',case,t,'A').integers(len(pool),size=250)]
        y=pool[rng('transfer',case,t,'B').integers(len(pool),size=250)]
        B=199;alpha=.05
    q=joint_test(x,y,g,rng('transfer',case,t,'permutation'),B=B)
    old=multiscale_test(x,y,rng('transfer',case,t,'old'),B=B,scales=(1.,)) if 'positive' in case else None
    r=dict(case=case,t=t,p=q['p'],reject=q['p']<=alpha,alpha=alpha,
           old_p=None if old is None else old['p_values']['old_hamming'],
           seconds=q['seconds'],B=B)
    atomic_json(path,r);return r

def replay():
    expected=read(BASE/'results/ucgnn-robust-distribution/freeze.json')['source_hashes']
    hashes={path:file_hash(path) for path in expected}
    assert hashes==expected
    items=[]
    for gid,label in [('regular-n512-d100','normalized-s41000'),('path3','normalized-s41000')]:
        g=next(g for g in read(GLOBAL/'results/ucgnn-global-distribution/graphs.json') if g['id']==gid)
        data=[]
        for who in ('Luby',label):
            with np.load(BASE/'data/ucgnn-robust-distribution/banks'/f'confirmatory__{who}__{gid}.npz') as a:
                data.append(a['colors'][:500][a['status'][:500]==0])
        old=read(BASE/'results/ucgnn-robust-distribution/confirmation-tests'/f'{gid}__{label}__m500.json')
        q=joint_test(*data,g,old_rng('confirmatory',gid,label,500,'omnibus'),B=19999,
                     futility=20 if old['p_kind']=='conservative_futility' else None)
        assert q['p']==old['p']
        assert q['statistics']==old['robust']['statistics']
        items.append(dict(graph=gid,model=label,p=q['p'],exact_statistics_match=True))
    return dict(source_hashes=hashes,replays=items)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    rep=replay();atomic_json(OUT/'frozen-v2-replay.json',rep)
    jobs=[(case,t) for case in ('teacher_empirical_null','neural_empirical_null','large_empirical_null') for t in range(200)]
    jobs += [('point_null',t) for t in range(20)]
    jobs += [(case,t) for case in ('star_positive','diffuse_positive') for t in range(30)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(trial,jobs))
    summary=[]
    for case in sorted(set(r['case'] for r in rows)):
        a=[r for r in rows if r['case']==case];k=sum(r['reject'] for r in a);N=len(a)
        ci=interval(k,N);flag=float(binomtest(k,N,.05,alternative='greater').pvalue)<.01/3 if 'null' in case else False
        summary.append(dict(case=case,trials=N,rejections=k,rate=k/N,ci95=ci,implementation_flag=flag,
                            old_rejections=sum(r['old_p']<=r['alpha'] for r in a) if 'positive' in case else None))
    passed=all(not r['implementation_flag'] for r in summary)
    passed &= all(r['ci95'][0]>=.8 for r in summary if 'positive' in r['case'])
    result=dict(passed=bool(passed),summary=summary,replay=rep,
                empirical_null_definition='Independent IID draws from the identical uniform distribution on the 32 completed pilot vectors. This is a finite empirical-law null, not a claim that a fitted law equals the true generator.',
                role='Transfer regression checks; full V2 calibration and blind spots remain as documented in the prior robust study.')
    atomic_json(OUT/'validation.json',result);print(json.dumps(result['summary']),flush=True)
    if not passed:raise RuntimeError('Transfer validation failed; no confirmation allowed')

if __name__=='__main__':main()
