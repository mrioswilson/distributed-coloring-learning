"""Independent fixed-size power at the primary family's strict tail."""
from concurrent.futures import ProcessPoolExecutor
from .common import *
from .controls import draw
from .stats import joint_test
from dcl.globaldist.analyze import wilson
import pandas as pd

def trial(job):
    case,m,t=job;path=OUT/'tail-validation'/f'{case}-{m}-{t}.json'
    if path.exists():return read(path)
    x,g,_=draw(case,m,rng('tail-validation2',case,m,t,'A'),0);y,_,_=draw(case,m,rng('tail-validation2',case,m,t,'B'),1)
    a=joint_test(x,y,g,rng('tail-validation2',case,m,t,'p'),B=19999,futility=20)
    r=dict(case=case,m=m,trial=t,p=a['p'],detect=bool(a['p']<=.05/795),permutations_used=a['permutations_used'],seconds=a['seconds'])
    atomic_json(path,r);return r
def main():
    jobs=[(c,m,t) for c,m in [('localized',250),('localized',500),('diffuse',500)] for t in range(30)]
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(trial,jobs))
    pd.DataFrame(rows).to_csv(OUT/'tail-validation-trials.csv',index=False);summary=[]
    for c,m in [('localized',250),('localized',500),('diffuse',500)]:
        a=[r for r in rows if r['case']==c and r['m']==m];k=sum(r['detect'] for r in a);ci=wilson(k,len(a))
        summary.append(dict(case=c,m=m,trials=len(a),detections=k,power=k/len(a),ci_low=ci[0],ci_high=ci[1],threshold=.05/795))
    atomic_json(OUT/'tail-validation.json',summary);print(summary,flush=True)
if __name__=='__main__':main()
