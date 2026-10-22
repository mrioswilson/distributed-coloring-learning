"""Exact-conditional count calibration; no chi-square approximation."""
import numpy as np

def likelihood_ratio(counts,totals,m,n):
    counts=np.atleast_2d(counts);other=totals[None,:]-counts
    ex=totals*m/(m+n);ey=totals*n/(m+n)
    return 2*np.sum(np.where(counts>0,counts*np.log(np.maximum(counts,1)/ex),0)+np.where(other>0,other*np.log(np.maximum(other,1)/ey),0),axis=1)

def categorical_test(x,y,generator,B=19999):
    z=np.r_[x,y];m,n=len(x),len(y);_,ids=np.unique(z,axis=0,return_inverse=True)
    totals=np.bincount(ids);observed=np.bincount(ids[:m],minlength=len(totals))
    stat=float(likelihood_ratio(observed,totals,m,n)[0]);ge=0
    for lo in range(0,B,1024):
        draws=generator.multivariate_hypergeometric(totals,m,size=min(1024,B-lo))
        ge+=int((likelihood_ratio(draws,totals,m,n)>=stat-1e-12).sum())
    return dict(p=(ge+1)/(B+1),statistic=stat,B=B,support_observed=len(totals),m=m,n=n,
                calibration='iid exact conditional multivariate-hypergeometric counts; plus one; inclusive ties')
