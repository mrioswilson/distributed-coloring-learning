"""Development-only relocation stress test; never confirmatory network data."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .common import *
from .controls import draw
from .stats import joint_test

def one(job):
    case,m,t=job;ids=(137,181) if case=='correlation' else (97,155,197)
    x,g,_=draw(case,m,rng('remote-development',case,m,t,'A'),0)
    y,_,_=draw(case,m,rng('remote-development',case,m,t,'B'),1)
    order=np.arange(g['n'])
    for old,new in zip(range(1,len(ids)+1),ids):order[old],order[new]=order[new],order[old]
    r=joint_test(x[:,order],y[:,order],g,rng('remote-development',case,m,t,'p'),B=199)
    return dict(case=case,m=m,trial=t,region=ids,p=r['p'])
def main():
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,[(c,m,t) for c in ('correlation','higher_order') for m in (250,1000) for t in range(10)]))
    atomic_json(OUT/'remote-development-v1.json',rows)
    for c in ('correlation','higher_order'):
        for m in (250,1000):print(c,m,sum(r['p']<=.05 for r in rows if r['case']==c and r['m']==m),'/10',flush=True)
if __name__=='__main__':main()
