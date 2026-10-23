"""Reported exploratory models; never used to choose the confirmatory test."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from .common import *

FORMS=('size','degree','ratio','density','additive','interaction')

def covariates(n,d,form):
    # Constants depend only on the fixed 4x4 design, not on outcomes.
    a=np.log2(n)-8.5;b=np.log2(d+1)-np.mean(np.log2(np.array(DS)+1))
    c=np.log(n/(d+1));density=np.log(d/(n-1))
    if form=='size':return [a]
    if form=='degree':return [b]
    if form=='ratio':return [c]
    if form=='density':return [density]
    if form=='additive':return [a,b]
    if form=='interaction':return [a,b,a*b]
    raise ValueError(form)

def fit(X,k,N):
    penalty=np.r_[np.full(6,1e-6),np.full(X.shape[1]-6,.5)]
    def objective(w):
        eta=X@w
        loss=np.sum(N*np.logaddexp(0,eta)-k*eta)+.5*np.sum(penalty*w*w)
        return float(loss),X.T@(N*expit(eta)-k)+penalty*w
    result=minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12})
    return result

def main():
    events=read(OUT/'elementary-events.json');bygraph={g['id']:g for g in graphs()};rows=[];models=[]
    for arch in ('normalized','deepsets','transformer'):
        labels=['Luby']+[s['label'] for s in specs() if s['treatment']==arch]
        for event in ('used_le_delta','uses_top_color'):
            data=[r for r in events if r['stratum']=='main' and r['model'] in labels and r['event']==event and r['total']>0]
            for form in FORMS:
                features=[]
                for r in data:
                    v=np.array(covariates(r['n'],r['delta'],form));is_model=r['model']!='Luby'
                    features.append(np.r_[np.eye(6)[labels.index(r['model'])],v*(not is_model),v*is_model])
                X=np.array(features);k=np.array([r['count'] for r in data]);N=np.array([r['total'] for r in data])
                groups=np.array([bygraph[r['graph']]['rep'] for r in data]);losses=[];converged=[]
                for holdout in range(3):
                    train=groups!=holdout;test=~train;f=fit(X[train],k[train],N[train]);eta=X[test]@f.x
                    loss=float(np.sum(N[test]*np.logaddexp(0,eta)-k[test]*eta)/N[test].sum())
                    rows.append(dict(architecture=arch,event=event,form=form,held_out_graph_rep=holdout,
                                     cross_entropy_per_run=loss,converged=bool(f.success),message=str(f.message)))
                    losses.append(loss);converged.append(bool(f.success))
                f=fit(X,k,N)
                models.append(dict(architecture=arch,event=event,form=form,mean_heldout_cross_entropy=float(np.mean(losses)),
                                   heldout_losses=losses,coefficient_labels=labels,coefficients=f.x.tolist(),
                                   converged=bool(f.success and all(converged)),penalty='intercepts 1e-6; slopes .5',
                                   scope='exploratory predictive binomial fits, fixed checkpoints, graph-replicate holdout; no coefficient p-values'))
    atomic_json(OUT/'exploratory-binomial-models.json',models)
    pd.DataFrame(rows).to_csv(OUT/'exploratory-prediction-folds.csv',index=False)
    pd.DataFrame(models).to_csv(OUT/'exploratory-prediction-summary.csv',index=False)
    print('Exploratory binomial prediction comparison complete',flush=True)

if __name__=='__main__':main()
