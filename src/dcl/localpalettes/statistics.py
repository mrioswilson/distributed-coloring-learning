"""Conditional-law distances and independent finite-sample tests."""
from __future__ import annotations

import numpy as np
from scipy.special import rel_entr
from scipy.stats import binom


def wilson(successes, n, confidence=.95):
    from scipy.stats import norm
    if n == 0:
        return [0., 1.]
    z = norm.ppf((1+confidence)/2)
    p, den = successes/n, 1+z*z/n
    center = (p+z*z/(2*n))/den
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0., center-radius), min(1., center+radius)]


def law_metrics(counts, available):
    counts = np.asarray(counts)
    n, k = counts.sum(), len(counts)
    q = counts/n
    target = np.zeros(k)
    target[available] = 1/len(available)
    middle = (q+target)/2
    entropy = -np.sum(q[q > 0]*np.log(q[q > 0]))
    tv = .5*np.abs(q-target).sum()
    radius = min(1., np.sqrt((k*np.log(2)+np.log(20))/(2*n)))
    allowed = np.zeros(k, bool)
    allowed[available] = True
    forbidden = int(counts[~allowed].sum())
    return dict(tv=float(tv), tv_lower=max(0., tv-radius), tv_upper=min(1., tv+radius),
                tv_radius=float(radius), equivalent_tv_005=bool(tv+radius <= .05),
                js=float(.5*(rel_entr(q,middle).sum()+rel_entr(target,middle).sum())),
                hellinger=float(np.sqrt(.5*np.sum((np.sqrt(q)-np.sqrt(target))**2))),
                affinity=float(np.clip(np.sqrt(q*target).sum(),0,1)), entropy=float(entropy),
                effective_entropy=float(np.exp(entropy)), effective_collision=float(1/(q*q).sum()),
                collision=float((q*q).sum()), max_mass=float(q.max()), forbidden_mass=float(forbidden/n),
                forbidden_interval=wilson(forbidden,n),
                forbidden_zero_upper=float(1-.05**(1/n)) if forbidden == 0 else None)


def log_weights(counts, available):
    q = (np.asarray(counts, dtype=float)+.5)/(np.sum(counts)+.5*len(counts))
    weights = np.full(len(counts), np.inf)
    weights[available] = np.log(q[available]*len(available))
    return weights


def likelihood_curve(counts, available, model_test, teacher_test, rng, calibration_trials=1024, tie_rng=None):
    weights = log_weights(counts, available)
    tie_rng = rng if tie_rng is None else tie_rng
    rows = []
    for m in (1,5,20,100,500,2000):
        null = weights[teacher_test[:, :m]].sum(1)
        alt = weights[model_test[:, :m]].sum(1)
        # Calibration draws only from the known teacher, separate from test.
        calibration = weights[rng.choice(available, size=(calibration_trials,m))].sum(1)
        threshold = float(np.quantile(calibration,.95,method='higher'))
        ties0, ties1 = tie_rng.integers(0,2,len(null)), tie_rng.integers(0,2,len(alt))
        correct0 = int(np.sum((null < 0) | ((null == 0) & (ties0 == 0))))
        correct1 = int(np.sum((alt > 0) | ((alt == 0) & (ties1 == 1))))
        fpr, power = int(np.sum(null > threshold)), int(np.sum(alt > threshold))
        i0, i1 = wilson(correct0,len(null),.975), wilson(correct1,len(alt),.975)
        rows.append(dict(m=m, balanced_accuracy=.5*(correct0/len(null)+correct1/len(alt)),
                         accuracy_lower=.5*(i0[0]+i1[0]), accuracy_upper=.5*(i0[1]+i1[1]),
                         power=power/len(alt), fpr=fpr/len(null),
                         power_interval=wilson(power,len(alt)), fpr_interval=wilson(fpr,len(null)),
                         threshold=threshold, calibration_fpr=float(np.mean(calibration>threshold)),
                         null_tie_rate=float(np.mean(null==0)), alt_tie_rate=float(np.mean(alt==0)),
                         trials_per_source=len(null)))
    return rows


def binary_product_tv(p, q, m):
    """Exact binomial sufficient-statistic TV for product Bernoulli laws."""
    x = np.arange(m+1)
    return float(.5*np.abs(binom.pmf(x,m,p)-binom.pmf(x,m,q)).sum())


def product_bounds(tv, affinity, m):
    return dict(lower=max(tv, 1-affinity**m),
                upper=min(1-(1-tv)**m, np.sqrt(max(0.,1-affinity**(2*m)))))


def uncertain_product_bounds(metric,m):
    """Valid on the event TV(Q,Qhat)<=r, using Hellinger triangle inequality."""
    r=metric['tv_radius']
    h=np.sqrt(max(0.,1-metric['affinity']))
    rho_lower=max(0.,1-min(1.,h+np.sqrt(r))**2)
    rho_upper=min(1.,1-max(0.,h-np.sqrt(r))**2)
    low=max(metric['tv_lower'],1-rho_upper**m)
    high=min(1-(1-metric['tv_upper'])**m,np.sqrt(max(0.,1-rho_lower**(2*m))))
    return dict(tv_lower=float(low),tv_upper=float(high),bayes_accuracy_lower=float(.5*(1+low)),
                bayes_accuracy_upper=float(.5*(1+high)),confidence=.95,
                note='Pointwise MC uncertainty propagated to true product laws; often uninformative')
