import json,time,math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from .common import *
from .stats import distance,u_stat,masks
from dcl.globaldist.core import seed as old_seed,holm
from dcl.globaldist.analyze import wilson

def replay(task):
    m,t=task;a=np.load(OLD/f'data/ucgnn-global-distribution/localized-positive-m{m}.npz')
    x,y=a['A'][t],a['B'][t];z=np.r_[x,y];K=np.exp(-distance(z,z));np.fill_diagonal(K,0)
    obs=float(u_stat(K,np.r_[np.ones(m),np.zeros(m)][None,:],m,m)[0])
    R=np.random.default_rng(old_seed('localized-control-perm',m,t));A=masks(2*m,m,199,R);null=u_stat(K,A,m,m)
    p=(1+int((null>=obs-1e-14).sum()))/200
    return dict(m=m,trial=t,p=p,statistic=obs,permutation_mean=float(null.mean()),permutation_sd=float(null.std(ddof=1)),
                q95=float(np.quantile(null,.95)),null=null.tolist() if t==0 else None)

def main():
    OUT.mkdir(parents=True,exist_ok=True);data=OLD/'results/ucgnn-global-distribution'
    rows=read(data/'primary.json');assert len(rows)==795
    assert np.allclose(holm([r['p'] for r in rows]),[r['p_holm'] for r in rows],atol=0,rtol=0)
    assert all(r['B']==19999 and r['gamma']==1 and r['m']==r['n']==250 for r in rows)
    assert sum(r['reject_holm'] for r in rows)==183
    p=pd.read_csv(data/'primary.csv');e=p[p.architecture=='normalized'].copy()
    st=pd.read_csv(data/'structural-comparisons.csv')
    e=e.merge(st,on=['graph','model'],how='left',suffixes=('','_structural'))
    power=pd.read_csv(data/'independent-power.csv');e['available_power_at_m250']=e.apply(lambda r: next((float(a.independent_population_power) for a in power.itertuples() if a.graph==r.graph and a.model==r.model and a.m==250),None),axis=1)
    e.to_csv(OUT/'old-e1-all-265.csv',index=False);e[e.reject_holm].to_csv(OUT/'old-e1-rejected-10.csv',index=False);e[~e.reject_holm].to_csv(OUT/'old-e1-not-rejected-255.csv',index=False)
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=[]
        for r in pool.map(replay,[(m,t) for m in (100,250,500,1000,2000) for t in range(20)]):
            results.append(r)
            if r['trial']==19:print('Replayed old star',r['m'],flush=True)
    archived=read(data/'localized-positive-control.json')
    assert all(r['p']==next(a for a in archived if a['m']==r['m'])['p_values'][r['trial']] for r in results)
    atomic_json(OUT/'old-star-replay.json',results)
    a=math.exp(-1/201);b=(1+a)/2;mu=b**200*(a+(1-a)/199)
    zeta=((1+a*a)/2)**200*(a*a+(1-a*a)/199)-mu*mu
    signal=b**200*(1-a)*(1/100-1/199)
    table=[]
    for m in (100,250,500,1000,2000):
        rs=[r for r in results if r['m']==m];k=sum(r['p']<=.05 for r in rs);sd=math.sqrt((4/(m*(m-1))+4/m**2)*zeta)
        table.append(dict(m=m,power=k/20,rejections=k,trials=20,ci95=wilson(k,20),exact_null_sd=sd,exact_signal=signal,signal_over_null_sd=signal/sd,
                          mean_observed_statistic=float(np.mean([r['statistic'] for r in rs])),mean_permutation_sd=float(np.mean([r['permutation_sd'] for r in rs]))))
    atomic_json(OUT/'old-statistical-audit.json',dict(passed=True,baseline_commit='1ba4546c64d143c44480644ebb9d61f9917c8982',historical_path=str(OLD),primary_hypotheses=795,
       holm_verified=True,B=19999,p_min=1/20000,strict_holm_threshold=.05/795,star_replay_exact_p_matches=100,star=table,star_exact_tv=99/199,star_exact_mmd2=signal,
       null_zeta2=zeta,approx_m_for_null_signal_to_sd_2_487=2.487*math.sqrt(8*zeta)/signal,seconds=time.monotonic()-start,
       original_source_hashes={str(p.relative_to(OLD)):file_hash(p) for p in (OLD/'src/dcl/globaldist').glob('*.py')}))
    text=f'''# Statistical audit of the original complete-coloring test

Verified actual worktree `{OLD}`, branch research/ucgnn-global-distribution,
commit 1ba4546. All original files were read only. The original 795 results
remain the baseline; they are not confirmatory data for this extension.

## Statistic, randomness and calibration

The kernel was exp(-d_H/n), gamma=1 fixed prospectively. A complete labeled
vector was one observation. Each algorithm run had a distinct per-run PCG64
stream, reset dynamic coloring/cache state, and independently generated
private participation/proposal uniforms. Teacher banks were shared only
across comparisons on the same graph, which creates dependence between tests
but does not invalidate Holm. Neural uniforms were explicitly converted to
float32; teacher uniforms remained float64. The audit checks original sample
and protocol metadata rather than treating vertices as independent replicates.

Permutation calibration used the unbiased two-sample MMD squared U statistic,
including negative values without truncation. The nonnegative biased norm
was separately reported as an effect, never used to replace U in the test.
The fixed family was all 53 graph entries times all 15 policies: 795 tests,
including two duplicate graph aliases with fresh banks (51 unique graphs).
All adjusted values were recomputed exactly and match the stored Holm values.
There were 183 rejections, including exactly 10 E1 cases, all retained here.

B=19,999 uniform fixed-size label permutations; p=(1+#{{T_perm >= T_obs-tol}})/(B+1).
Inclusive ties with tolerance 1e-14 at the primary size were conservative.
The final reusable helper decreases tolerance above N=500; the historical
star supplement used constant 1e-14, which this audit reconstructs explicitly.
Minimum p={1/20000}, smaller than .05/795={.05/795:.9g}. Thus the resolution
was sufficient. Many p-values saturated at that minimum: finer ranking was
unavailable. No parametric tail extrapolation was used. The 795 stored
individual results have the expected independent samples, gamma and B.

Each coordinate kernel matrix is (1-a)I+a11^T with a=exp(-1/n), strictly
positive definite. Its finite product and restriction to proper colorings
remain strictly positive definite, hence characteristic. This proves
identifiability and asymptotic consistency, not finite-sample sensitivity.

## Exact reproduction and quantitative failure of the star control

On the 201-vertex star, leaves are iid uniform {{0,1}} under both laws.
The independent center is uniform {{2,...,200}} under A and {{2,...,101}}
under B. Both use valid local lists. TV=99/199. Let a=exp(-1/201), b=(1+a)/2.
The exact kernel effect is b^200(1-a)(1/100-1/199)={signal:.12g}.
The large center alphabet and irrelevant leaves attenuate the signal; the
second center support is a subset of the first, not disjoint from it.

Under A/A, the kernel row expectation is constant mu=b^200[a+(1-a)/199].
For the centered kernel, zeta2=E[k^2]-mu^2={zeta:.12g}, with
E[k^2]=((1+a^2)/2)^200[a^2+(1-a^2)/199]. For equal sample sizes the exact
null variance of U is [4/(m(m-1))+4/m^2] zeta2. This provides the signal/noise
comparison below independently of Monte Carlo estimates.

| m | Rejections/20 | Exact null SD | Signal/null SD |
|---:|---:|---:|---:|
'''
    for r in table:text+=f"| {r['m']} | {r['rejections']}/20 | {r['exact_null_sd']:.7g} | {r['signal_over_null_sd']:.4f} |\n"
    text+='''
All 100 historical p-values were reproduced exactly from archived vectors
and named permutation seeds. Full observed statistics, first-trial null
distributions, quantiles and binomial intervals are archived. The control
used B=199, nominal alpha=.05, without external Holm. Thus its failure cannot
be blamed on the 795-test correction. Leaves contribute kernel variability
although their laws coincide; Hamming normalization and high center support
jointly make the mean effect small. Increasing bandwidth globally may also
suppress similarity between noisy leaf vectors; it does not isolate the hub.
For analogous growing stars the effect is O(D^-2), while null SD is
O(1/(m sqrt(D))). Characteristicness supplies no useful dimension-free TV
lower bound. A normal signal/SD planning approximation is recorded as a
diagnostic only, never used to calculate p-values or claim guaranteed power.

## E1 inventory and limits

old-e1-all-265.csv retains graph/family/actual degree/size/checkpoint, U,
raw and adjusted p, graph and checkpoint hashes, structural projections,
and available independent m=250 power (only seed 41000 on selected graphs).
Separate ten-rejection and 255-nonrejection files preserve both outcomes.
Missing power is missing, not zero. No nonrejection is interpreted as equality.
The previous conservative MMD intervals establish no TV bound or approximate
equivalence. The initial old resource forecast lacked a per-cell archived
table; realized costs and the original budget ledger remain available.
The permutation and Holm implementations show no identified inflation in
this audit. Their finite controls do not certify numerical behavior for all
inputs; the observed principal methodological defect is power for localized
alternatives, not insufficient primary permutation resolution.
'''
    Path('docs/OLD_TEST_STATISTICAL_AUDIT.md').write_text(text)
    print('Old statistical audit complete',flush=True)
if __name__=='__main__':main()
