"""Whole-run events and fixed-cohort factorial inference.

Shared Luby banks enter each linear contrast once with the combined weight.
No vertex, graph, or checkpoint is counted as an independent coloring run.
"""
import numpy as np
from scipy.stats import fisher_exact
from dcl.robustdist.classifier import interval

EVENTS=('used_le_delta','uses_top_color')
STRUCTURAL=('used_colors','maximum_color','palette_fraction','largest_class_fraction',
            'entropy_normalized','mean_color_normalized','nonedge_reuse')

def measurements(z,g):
    z=np.asarray(z);n=g['n'];K=g['delta']+1
    counts=np.array([np.bincount(row,minlength=K) for row in z])
    used=(counts>0).sum(1);p=counts/n
    nonedges=n*(n-1)-len(g['edges'][0])
    return dict(used_colors=used,maximum_color=z.max(1),
                used_le_delta=used<=g['delta'],uses_top_color=counts[:,-1]>0,
                palette_fraction=used/K,largest_class_fraction=p.max(1),
                entropy_normalized=-(p*np.log(np.maximum(p,1e-100))).sum(1)/np.log(K),
                mean_color_normalized=(z/np.maximum(np.array(g['degrees']),1)).mean(1),
                nonedge_reuse=(counts*(counts-1)).sum(1)/max(nonedges,1),
                color_counts=counts)

def event_difference(kx,nx,ky,ny,alpha):
    if not nx or not ny:
        return dict(probability_Luby=None,probability_model=None,difference=None,ci=[-1,1],tv_lower=0.,p=1.)
    cx=interval(kx,nx,alpha);cy=interval(ky,ny,alpha)
    ci=[cy[0]-cx[1],cy[1]-cx[0]]
    return dict(probability_Luby=kx/nx,probability_model=ky/ny,difference=ky/ny-kx/nx,
                ci=ci,tv_lower=max(0.,ci[0],-ci[1]),ci_Luby=cx,ci_model=cy,
                p=float(fisher_exact([[kx,nx-kx],[ky,ny-ky]]).pvalue))

def linear_interval(rows,weights,alpha_hoeffding,cp_alpha):
    """Independent bank-wise binomial model, conditional on valid counts.

    Rows have count, total. Weights are summed before applying concentration;
    reuse of the same teacher is therefore never mistaken for new data.
    CP simultaneous coverage is supplied by cp_alpha over all elementary p's.
    The caller allocates the two bounds' total failure budgets, then intersects.
    """
    active=[(rows[k],float(w)) for k,w in weights.items() if abs(w)>1e-15]
    if any(r['total']==0 for r,w in active):
        return dict(estimate=None,ci=[-sum(abs(w) for r,w in active),sum(abs(w) for r,w in active)],significant=False)
    estimate=sum(w*r['count']/r['total'] for r,w in active)
    lo=hi=0.;scale=0.
    for r,w in active:
        a,b=interval(r['count'],r['total'],cp_alpha)
        lo+=w*(a if w>=0 else b);hi+=w*(b if w>=0 else a)
        scale+=w*w/r['total']
    radius=np.sqrt(.5*scale*np.log(2/alpha_hoeffding))
    ci=[max(lo,estimate-radius),min(hi,estimate+radius)]
    # Conflicting bounds are a possible noncoverage realization, never hidden.
    return dict(estimate=float(estimate),ci=list(map(float,ci)),cp_ci=[float(lo),float(hi)],
                hoeffding_radius=float(radius),significant=bool(ci[0]>0 or ci[1]<0),
                bounds_conflict=bool(ci[0]>ci[1]))

def cell_weights(gs,labels,coefficients):
    out={}
    for (n,d),a in coefficients.items():
        selected=[g for g in gs if g['stage']=='main' and g['n']==n and g['delta']==d]
        if not selected:raise ValueError('Missing frozen cell')
        for g in selected:
            for label in labels:
                key=(g['id'],label);out[key]=out.get(key,0.)+a/(len(selected)*len(labels))
            key=(g['id'],'Luby');out[key]=out.get(key,0.)-a/len(selected)
    return out

def contrast_specs(ns,ds):
    out=[]
    for n in ns:
        for d in ds:out.append(('cell',f'n{n}-d{d}',{(n,d):1}))
    for d in ds:
        for a,b in zip(ns[:-1],ns[1:]):
            out.append(('size',f'n{b}-n{a}|d{d}',{(b,d):1,(a,d):-1}))
    for n in ns:
        for a,b in zip(ds[:-1],ds[1:]):
            out.append(('degree',f'd{b}-d{a}|n{n}',{(n,b):1,(n,a):-1}))
    for na,nb in zip(ns[:-1],ns[1:]):
        for da,db in zip(ds[:-1],ds[1:]):
            out.append(('interaction',f'n{na}:{nb}-d{da}:{db}',{(nb,db):1,(na,db):-1,(nb,da):-1,(na,da):1}))
    out.append(('interaction','full-corners',{(ns[-1],ds[-1]):1,(ns[0],ds[-1]):-1,(ns[-1],ds[0]):-1,(ns[0],ds[0]):1}))
    return out
