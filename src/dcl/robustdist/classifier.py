"""Non-neural classifier two-sample test with independent validation/test."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import hypergeom,beta
from .stats import fit_features,evaluate

def interval(k,n,alpha=.05):
    return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]

def classifier_test(x,y,g):
    cut=len(x)//2;t=max(2,3*cut//4)
    a,b=x[:t],y[:t];desc=fit_features(a,b,g)
    train=evaluate(desc,np.r_[a,b],g);val=evaluate(desc,np.r_[x[t:cut],y[t:cut]],g)
    test=evaluate(desc,np.r_[x[cut:],y[cut:]],g)
    mean=train.mean(0);sd=np.maximum(train.std(0),1e-6)
    train=(train-mean)/sd;val=(val-mean)/sd;test=(test-mean)/sd
    train=np.c_[np.ones(len(train)),train];val=np.c_[np.ones(len(val)),val];test=np.c_[np.ones(len(test)),test]
    labels=np.r_[np.ones(t),np.zeros(t)];vl=np.r_[np.ones(cut-t),np.zeros(cut-t)]
    candidates=[]
    for penalty in (.1,1.,10.):
        def fun(w):
            eta=train@w
            return float(np.logaddexp(0,eta).sum()-labels@eta+penalty*np.sum(w[1:]**2)/2),train.T@(expit(eta)-labels)+penalty*np.r_[0.,w[1:]]
        w=minimize(fun,np.zeros(train.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':100}).x
        candidates.append((float(np.mean((val@w>0)==vl)),f'logistic_L2_{penalty}',test@w>0))
    mus=[train[labels==c].mean(0) for c in (0,1)];vars=[train[labels==c].var(0)+.1 for c in (0,1)]
    def score(z,c):return -.5*(np.log(vars[c])+(z-mus[c])**2/vars[c]).sum(1)
    candidates.append((float(np.mean((score(val,1)>score(val,0))==vl)),'GaussianNB',score(test,1)>score(test,0)))
    accuracy,name,pred=max(candidates,key=lambda a:a[0]);m=len(x)-cut;n=len(y)-cut
    tp=int(pred[:m].sum());tn=int((~pred[m:]).sum());K=int(pred.sum())
    ci0=interval(tp,m,.025);ci1=interval(tn,n,.025)
    # Balanced-size exact conditional permutation distribution of predictions.
    p=float(hypergeom.sf(tp-1,m+n,K,m))
    return dict(method=name,validation_accuracy=accuracy,balanced_accuracy=(tp/m+tn/n)/2,p=p,
                ci95=[(ci0[0]+ci1[0])/2,(ci0[1]+ci1[1])/2],train_per_class=t,validation_per_class=cut-t,evaluation_per_class=m)
