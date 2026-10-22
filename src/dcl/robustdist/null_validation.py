"""Fresh actual-protocol self controls, independent of synthetic development."""
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import time,argparse
import numpy as np
import pandas as pd
from .common import *
from .stats import joint_test,multiscale_test,multinomial_test
from .classifier import classifier_test
from dcl.globaldist.analyze import wilson
from scipy.stats import binom

@lru_cache(None)
def bank(file):
    a=np.load(file);assert not a['status'].any()
    return a['colors']

def trial(job):
    label,file,t,phase=job;path=OUT/f'{phase}-trials'/f'{label}-{t}.json'
    if path.exists():return read(path)
    z=bank(file);x,y=z[500*t:500*t+250],z[500*t+250:500*t+500];g=next(g for g in graphs() if g['id']=='path3')
    a=joint_test(x,y,g,rng(phase,label,t,'omni'),B=199)
    b=multiscale_test(x,y,rng(phase,label,t,'global'),B=199)
    c=classifier_test(x,y,g);d=multinomial_test(x,y,rng(phase,label,t,'direct'),B=199)
    r=dict(label=label,trial=t,m=250,source_file=file,p_values={**a['p_values'],**b['p_values'],'classifier':c['p'],'multinomial':d['p']})
    atomic_json(path,r);return r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--phase',default='null');args=parser.parse_args()
    q=DATA/f'queue-{args.phase}.json'
    while not q.exists() or not all(t['status']=='complete' for t in read(q)['tasks']):
        if not args.watch:raise RuntimeError('Actual null banks incomplete')
        time.sleep(10)
    jobs=[]
    for t in read(q)['tasks']:
        label='Luby' if t['model'] is None else t['model']['label']
        jobs.extend((label,t['file'],i,args.phase) for i in range(400))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(trial,jobs),1):
            rows.extend(dict(label=r['label'],trial=r['trial'],method=method,p=p,reject=p<=.05) for method,p in r['p_values'].items())
            if i%100==0:print('Actual null validation',i,len(jobs),flush=True)
    suffix='2' if args.phase=='null2' else ''
    pd.DataFrame(rows).to_csv(OUT/f'actual-null-trials{suffix}.csv',index=False);summary=[]
    for (label,method),a in pd.DataFrame(rows).groupby(['label','method']):
        k=int(a.reject.sum());n=len(a);ci=wilson(k,n)
        summary.append(dict(label=label,method=method,trials=n,rejections=k,rate=k/n,ci_low=ci[0],ci_high=ci[1],null_exceedance_p=float(binom.sf(k-1,n,.05))))
    atomic_json(OUT/f'actual-null-validation{suffix}.json',summary);pd.DataFrame(summary).to_csv(OUT/f'actual-null-validation{suffix}.csv',index=False)
    print('Actual null validation complete',flush=True)
if __name__=='__main__':main()
