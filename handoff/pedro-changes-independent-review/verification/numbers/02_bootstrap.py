# Recompute paired-seed bootstrap from raw confusion counts (seedtable.json from 01).
import json,sys,numpy as np, itertools, math
T=json.load(open(sys.argv[1]))
seeds=range(105001,105011)
def diffs(t):
    return np.array([T[f'{t}|{s}|cf-test']['ba']-T[f'{t}|{s}|native-test']['ba'] for s in seeds])
def interval(d,rng,B=10000):
    samples=d[rng.integers(len(d),size=(B,len(d)))].mean(1)
    return np.quantile(samples,[.025,.975])
be=diffs('be_base_d3'); gm=diffs('speculative_gm')
print('BE native mean BA',np.mean([T[f'be_base_d3|{s}|native-test']['ba'] for s in seeds])*100)
print('BE cf mean BA',np.mean([T[f'be_base_d3|{s}|cf-test']['ba'] for s in seeds])*100)
print('BE diffs pp',np.round(be*100,3),'mean',be.mean()*100, 'median',np.median(be)*100)
# Pedro's exact procedure: rng seed 2026103017, BE first then GM with same generator
rng=np.random.default_rng(2026103017)
ibe=interval(be,rng); igm=interval(gm,rng)
print('Pedro-procedure BE CI',ibe*100,'GM CI',igm*100)
# sensitivity across 200 rng seeds
lo=[];hi=[]
for k in range(200):
    r=np.random.default_rng(k); i=interval(be,r); lo.append(i[0]);hi.append(i[1])
print('BE CI over 200 rng seeds: lo range',min(lo)*100,max(lo)*100,' hi range',min(hi)*100,max(hi)*100)
# exact bootstrap distribution via multinomial enumeration (n=10: C(19,9)=92378 compositions)
n=10; vals=be
from math import factorial
probs=[];means=[]
def comps(n,k):
    for c in itertools.combinations(range(n+k-1),k-1):
        prev=-1; out=[]
        for x in c: out.append(x-prev-1); prev=x
        out.append(n+k-1-prev-1); yield out
for c in comps(n,n):
    p=factorial(n)
    for x in c: p//=factorial(x)
    probs.append(p/ n**n); means.append(sum(ci*v for ci,v in zip(c,vals))/n)
o=np.argsort(means); m=np.array(means)[o]; pr=np.cumsum(np.array(probs)[o])
q=lambda a: m[np.searchsorted(pr,a)]
print('exact bootstrap 2.5/97.5% quantiles',q(.025)*100,q(.975)*100,'sum p',pr[-1])
# t interval
sd=be.std(ddof=1); from statistics import NormalDist
tcrit=2.2621571628  # t_{0.975,9}
print('t-interval',(be.mean()-tcrit*sd/math.sqrt(10))*100,(be.mean()+tcrit*sd/math.sqrt(10))*100, 'sd',sd*100)
# sign/Wilcoxon-ish: count positive
print('positive diffs',int((be>0).sum()),'negative',int((be<0).sum()))
# exact sign-flip permutation test of mean diff (paired randomization)
cnt=0;tot=0
for signs in itertools.product([1,-1],repeat=10):
    tot+=1; cnt+= abs((np.array(signs)*be).mean())>=abs(be.mean())-1e-15
print('exact sign-flip p (two-sided)',cnt/tot)
# BCa not done; also eq-type paired diffs
et=np.array([T[f'be_base_d3|{s}|cf-test']['eqtype']-T[f'be_base_d3|{s}|native-test']['eqtype'] for s in seeds])
print('BE eqtype paired diffs pp',np.round(et*100,3),'mean',et.mean()*100,'min',et.min()*100,'all positive',bool((et>0).all()))
rng=np.random.default_rng(1); print('eqtype bootstrap CI',interval(et,rng)*100)
cnt=0
for signs in itertools.product([1,-1],repeat=10):
    cnt+= abs((np.array(signs)*et).mean())>=abs(et.mean())-1e-15
print('eqtype sign-flip p',cnt/1024)
gt=np.array([T[f'speculative_gm|{s}|cf-test']['eqtype']-T[f'speculative_gm|{s}|native-test']['eqtype'] for s in seeds])
print('GM eqtype diffs',np.round(gt*100,3),'means native/cf',np.mean([T[f'speculative_gm|{s}|native-test']['eqtype'] for s in seeds])*100)
# success-weighted sums
for t in ['be_base_d3','speculative_gm']:
    print(t,'succ N',sum(T[f'{t}|{s}|native-test']['succ'] for s in seeds),'CF',sum(T[f'{t}|{s}|cf-test']['succ'] for s in seeds))
    print(t,'mean steps N',np.mean([T[f'{t}|{s}|native-test']['mean_steps'] for s in seeds]),'CF',np.mean([T[f'{t}|{s}|cf-test']['mean_steps'] for s in seeds]))
    for arm in ['native-test','cf-test']:
        fs=sum(T[f'{t}|{s}|{arm}']['fail_steps_sum'] for s in seeds); nf=sum(T[f'{t}|{s}|{arm}']['n_fail'] for s in seeds)
        ss=set().union(*[set(T[f'{t}|{s}|{arm}']['succ_steps']) for s in seeds])
        print('  ',arm,'failures',nf,'mean fail steps',fs/nf if nf else None,'success step values',ss)
        kinds={}
        for s in seeds:
            for k,v in T[f'{t}|{s}|{arm}']['fail_kinds'].items(): kinds[k]=kinds.get(k,0)+v
        print('    fail kinds',kinds)
# identical failure check 105010
a=T['be_base_d3|105010|native-test']; b=T['be_base_d3|105010|cf-test']
print('105010 identical correct vectors',a['correct_vec']==b['correct_vec'],'same case ids',a['case_ids']==b['case_ids'])
for s in seeds:
    a=T[f'be_base_d3|{s}|native-test']; b=T[f'be_base_d3|{s}|cf-test']
    print(s,'BE same case_ids',a['case_ids']==b['case_ids'],'same fingerprints count', sum(x==y for x,y in zip(a['fingerprints'],b['fingerprints'])))
