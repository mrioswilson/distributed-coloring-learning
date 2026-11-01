# Recompute historical BE type-weighting table (revision-type-analysis.tex) from be-type-counts.json
import json,sys
from collections import defaultdict
rows=json.load(open(sys.argv[1]))
print('methods',sorted(set(r['method'] for r in rows)),'seeds',sorted(set(r['seed'] for r in rows)))
by=defaultdict(list)
for r in rows: by[(r['method'],r['seed'])].append(r)
def metrics(rs,predfun=None):
    tp=tn=fp=fn=0; tacc=[]; n=0; corr=0
    for r in rs:
        o=r['occurrences']; y=r['truth']
        c=r['correct'] if predfun is None else (o if predfun(r)==y else 0)
        n+=o; corr+=c; tacc.append(c/o)
        if y: tp+=c; fn+=o-c
        else: tn+=c; fp+=o-c
    return corr/n, .5*(tp/(tp+fn)+tn/(tn+fp)), sum(tacc)/len(tacc), n, len(rs)
seeds=sorted(set(r['seed'] for r in rows))
def elig(r):
    k=r['key']; return k[2]==1 and k[3]>=0 and k[4][2]==1
for m in sorted(set(r['method'] for r in rows)):
    M=[metrics(by[(m,s)]) for s in seeds]
    print(m,'occ %.3f BA %.3f type %.3f'%tuple(100*sum(x[i] for x in M)/len(M) for i in range(3)),'n',[x[3] for x in M],'types',[x[4] for x in M])
ref=sorted(set(r['method'] for r in rows))[0]
for name,f in (('always_reject',lambda r:False),('eligibility_only',elig)):
    M=[metrics(by[(ref,s)],f) for s in seeds]
    print(name,'occ %.3f BA %.4f type %.3f'%tuple(100*sum(x[i] for x in M)/len(M) for i in range(3)))
# BA of CF to 4 decimals
for m in sorted(set(r['method'] for r in rows)):
    M=[metrics(by[(m,s)]) for s in seeds]; print(m,'BA 6dp %.6f occ 6dp %.6f'%(100*sum(x[1] for x in M)/5,100*sum(x[0] for x in M)/5))
