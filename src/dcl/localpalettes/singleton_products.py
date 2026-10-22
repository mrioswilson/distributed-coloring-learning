"""Exact product-law form for a singleton teacher; estimated effects retain CIs."""
import json
import numpy as np
from scipy.stats import beta

from .config import ROOT, RESULTS
from .report import write_csv


def product_tv(forbidden_mass,m):
    if forbidden_mass>=1:return 1.
    return float(-np.expm1(m*np.log1p(-forbidden_mass)))


def sample_size(forbidden_mass,target=.8):
    if forbidden_mass<=0:return None
    if forbidden_mass>=1:return 1
    return int(np.ceil(np.log1p(-target)/np.log1p(-forbidden_mass)))


def analyze():
    states=sum((json.loads((RESULTS/name).read_text()) for name in
                ('state-manifest.json','autonomous-state-manifest.json','historical-state-manifest.json')),[])
    single={s['id']:s for s in states if len(s['available'])==1}
    paths=[p for p in sorted((ROOT/'distributions').rglob('*.json')) if p.stem in single]
    family_size=len(paths);alpha=.05/max(1,family_size);rows=[]
    for path in paths:
        r=json.loads(path.read_text())['metrics'];s=single[r['state']];x=np.load(path.with_suffix('.npz'))
        counts=x['estimation_counts'];test=x['test_counts'];a=s['available'][0]
        n=int(counts.sum());bad=int(n-counts[a]);effect=bad/n
        lower=0. if bad==0 else float(beta.ppf(alpha/2,bad,n-bad+1))
        upper=1. if bad==n else float(beta.ppf(1-alpha/2,bad+1,n-bad))
        for m in (1,5,20,100,500,2000):
            power=product_tv(effect,m)
            rows.append(dict(model=r['model'],state=r['state'],k=r['k'],m=m,forbidden_mass=effect,
                             forbidden_lower=lower,forbidden_upper=upper,analytic_power_estimate=power,
                             analytic_power_lower=product_tv(lower,m),analytic_power_upper=product_tv(upper,m),
                             bayes_accuracy_estimate=.5*(1+power),
                             bayes_accuracy_lower=.5*(1+product_tv(lower,m)),bayes_accuracy_upper=.5*(1+product_tv(upper,m)),
                             false_positive_rate=0.,m80_estimate=sample_size(effect),m80_lower=sample_size(upper),m80_upper=sample_size(lower),
                             independent_m1_power=float((test.sum()-test[a])/test.sum()) if m==1 else None,
                             confidence='95% simultaneous Bonferroni Clopper-Pearson over singleton model-state laws',
                             confidence_family_size=family_size,
                             interpretation='Post-hoc analytic product-law prediction; no new m-sample test bank'))
    write_csv(RESULTS/'singleton-product-laws.csv',rows)
    print(f'Singleton analytic product laws: {len(rows)} rows from {family_size} independently estimated laws.')
    return rows


if __name__=='__main__':analyze()
