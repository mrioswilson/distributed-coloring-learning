"""Resumable CPU-only benchmark with distinct development/validation seeds."""
import argparse,time,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.stats import binom
from .common import *
from .controls import draw,CASES
from .stats import joint_test,multiscale_test,multinomial_test
from .classifier import classifier_test
from dcl.globaldist.analyze import wilson

def trial(job):
    stage,case,m,t,D,effect=job;ident=f'{stage}-{case}-D{D}-e{effect}-m{m}-t{t}'
    path=OUT/'benchmark-trials'/f'{ident}.json'
    if path.exists():return read(path)
    start=time.monotonic();x,g,meta=draw(case,m,rng(stage,case,m,t,D,effect,'A'),0,D,effect)
    y,_,_=draw(case,m,rng(stage,case,m,t,D,effect,'B'),1,D,effect)
    assert valid(x,g).all() and valid(y,g).all()
    raw=DATA/'benchmark'/f'{ident}.npz';save_bank(raw,X=x,Y=y)
    r=joint_test(x,y,g,rng(stage,case,m,t,D,effect,'omni'),B=199)
    q=multiscale_test(x,y,rng(stage,case,m,t,D,effect,'multi'),B=199)
    c=classifier_test(x,y,g);direct=multinomial_test(x,y,rng(stage,case,m,t,D,effect,'direct'),B=199)
    result=dict(id=ident,stage=stage,case=case,m=m,trial=t,D=D,effect=effect,**meta,
                p_values={**r['p_values'],**q['p_values'],'classifier':c['p'],'multinomial':direct['p']},
                omnibus_statistics=r['statistics'],global_statistics=q['statistics'],classifier=c,
                seconds=time.monotonic()-start,omnibus_seconds=r['seconds'],quadratic_seconds=q['seconds'],raw_file=str(raw),raw_sha256=file_hash(raw))
    atomic_json(path,result);return result

def plan(stage):
    if stage=='pilot':return [(stage,'localized',m,0,200,None) for m in (100,250,500,1000,2000)]
    if stage=='development':return [(stage,c,m,t,200,None) for c in CASES[4:] for m in (100,250) for t in range(10)]
    if stage=='validation':
        return [(stage,c,250,t,200,None) for c in CASES[:4] for t in range(400)]+[(stage,c,m,t,200,None) for c in CASES[4:] for m in (100,250,500,1000,2000) for t in range(30)]
    if stage=='sensitivity':
        cells=[('localized',D,None) for D in (20,1000)]+[('marginal',200,p) for p in (.6,.7)]+[('diffuse',200,p) for p in (.51,.53)]+[('rare',200,.01)]
        return [(stage,c,m,t,D,e) for c,D,e in cells for m in (250,1000) for t in range(30)]
    raise ValueError(stage)

def summarize(stage):
    records=[]
    for f in (OUT/'benchmark-trials').glob(stage+'-*.json'):
        r=read(f)
        for method,p in r['p_values'].items():records.append({k:r[k] for k in ('stage','case','m','trial','D','effect','tv','seconds','omnibus_seconds','quadratic_seconds')}|dict(method=method,p=p,reject=p<=.05))
    frame=pd.DataFrame(records);frame.to_csv(OUT/f'benchmark-{stage}-trials.csv',index=False)
    rows=[]
    for ids,a in frame.groupby(['case','m','D','effect','method'],dropna=False):
        k=int(a.reject.sum());n=len(a);lo,hi=wilson(k,n)
        rows.append(dict(zip(('case','m','D','effect','method'),ids))|dict(trials=n,rejections=k,rate=k/n,ci_low=lo,ci_high=hi,mean_wall_seconds=float(a.seconds.mean()),null_exceedance_p=float(binom.sf(k-1,n,.05))))
    pd.DataFrame(rows).to_csv(OUT/f'benchmark-{stage}.csv',index=False)
    atomic_json(OUT/f'benchmark-{stage}.json',rows)
    return frame,rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','development','validation','sensitivity']);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);jobs=plan(args.stage);begin=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(trial,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result()
            if args.stage=='pilot' or i%25==0:print(json.dumps(dict(complete=i,total=len(jobs),last=r['id'],last_seconds=r['seconds'],elapsed=time.monotonic()-begin)),flush=True)
    summarize(args.stage)
    print('Benchmark complete',args.stage,len(jobs),flush=True)
if __name__=='__main__':main()
