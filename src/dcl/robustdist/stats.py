"""Each row is an observation; vertex columns are never permuted.

Feature selection uses only the first half of each independent sample. All
evaluation-label permutations keep selection fixed. Pooled scales ignore
evaluation labels. A full-vector characteristic kernel is always retained.
"""
import itertools,time
import numpy as np
from scipy.special import logsumexp
from dcl.globaldist.core import holm
from .common import rng

SCALES=(.25,1.,4.,16.)

def distance(x,y):
    out=np.zeros((len(x),len(y)),float)
    for v in range(x.shape[1]):out+=x[:,v,None]!=y[None,:,v]
    return out/x.shape[1]

def masks(N,m,B,generator):
    A=np.zeros((B,N),float)
    for i in range(B):A[i,generator.choice(N,m,replace=False)]=1
    return A

def u_stat(K,A,m,n):
    """K has zero diagonal. Matches direct unbiased MMD, including negatives."""
    AK=A@K;aa=np.sum(AK*A,axis=1);ar=AK.sum(1)
    return aa/(m*(m-1))+(K.sum()-2*ar+aa)/(n*(n-1))-2*(ar-aa)/(m*n)

def kernel_scale(K,m,n):
    H=K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean()
    return max(float(np.linalg.norm(H))*np.sqrt(2)*(1/m+1/n)/len(K),1e-15)

def groups(g):
    n=g['n'];degree=np.array(g['degrees']);r=rng('feature-regions',g['id'])
    pairs=set(itertools.combinations(range(min(n,10)),2))
    if n>1:
        for _ in range(64):pairs.add(tuple(sorted(r.choice(n,2,replace=False))))
    adj=[[] for _ in range(n)]
    for a,b in np.array(g['edges'],int).reshape(2,-1).T:adj[a].append(int(b))
    for v in np.argsort(-degree,kind='stable')[:4]:
        for a,b in itertools.combinations(sorted(adj[v])[:8],2):pairs.add((a,b))
    triples=set(itertools.combinations(range(min(n,7)),3))
    if n>=3:
        for _ in range(32):triples.add(tuple(sorted(r.choice(n,3,replace=False))))
    return sorted(pairs),sorted(triples)

def structural(z,g):
    n=g['n'];K=max(g['degrees'])+1
    counts=np.zeros((len(z),K),float)
    for i,row in enumerate(z):counts[i]=np.bincount(row,minlength=K)
    used=(counts>0).sum(1);p=counts/n
    features=[used,counts.max(1)/n,-(p*np.log(np.maximum(p,1e-100))).sum(1),z.mean(1)]
    names=['used_colors','largest_class_fraction','class_entropy','mean_color']
    pairs,_=groups(g)
    edge=set(map(tuple,np.array(g['edges'],int).reshape(2,-1).T.tolist()))
    for i,(a,b) in enumerate(pairs):
        if (a,b) not in edge:features.append((z[:,a]==z[:,b]).astype(float));names.append(f'cocolor:{a},{b}')
    for d in sorted(set(g['degrees'])):
        ids=np.flatnonzero(np.array(g['degrees'])==d)
        features.extend([z[:,ids].mean(1),np.sum(z[:,ids]%2,axis=1)%2])
        names.extend([f'degree_mean:{d}',f'degree_parity:{d}'])
    return np.column_stack(features),names

def fit_features(x,y,g):
    """Fit bounded signed categorical witnesses and select by training signal.

    The returned descriptors suffice to evaluate on a separate holdout. No
    evaluation labels, permutations, or p-values enter this function.
    """
    z=np.r_[x,y];m=len(x);descriptors=[];pools={k:[] for k in ('local','interaction','structural')}
    pairs,triples=groups(g)
    region_list=[('local',(v,)) for v in range(g['n'])]+[('interaction',p) for p in pairs+triples]
    dense=[]
    for family,ids in region_list:
        keys,inv=np.unique(z[:,ids],axis=0,return_inverse=True)
        a=np.bincount(inv[:m],minlength=len(keys))/m;b=np.bincount(inv[m:],minlength=len(keys))/len(y)
        score=(a-b)/(a+b+1/m)
        vals=score[inv];sd=max(vals.std(),1e-8)
        signal=abs(vals[:m].mean()-vals[m:].mean())/sd
        desc=dict(kind='categorical',family=family,ids=list(map(int,ids)),keys=keys.tolist(),scores=score.tolist())
        pools[family].append((signal,desc))
        if len(ids)==1:
            dense.append(desc)
            ordered=z[:,ids[0]];values=np.unique(ordered)
            if len(values)>1:
                diffs=np.array([(ordered[:m]<=v).mean()-(ordered[m:]<=v).mean() for v in values[:-1]])
                j=int(np.argmax(abs(diffs)));v=int(values[j]);vals=(ordered<=v).astype(float)
                signal=abs(diffs[j])/max(vals.std(),1e-8)
                pools[family].append((signal,dict(kind='threshold',family=family,ids=[int(ids[0])],threshold=v)))
    s,names=structural(z,g)
    for j,name in enumerate(names):
        vals=s[:,j];signal=abs(vals[:m].mean()-vals[m:].mean())/max(vals.std(),1e-8)
        pools['structural'].append((signal,dict(kind='structural',family='structural',column=j,name=name)))
        values=np.unique(vals)
        if len(values)>1:
            thresholds=np.unique(np.quantile(values,np.linspace(0,1,17))[:-1])
            diffs=np.array([(vals[:m]<=v).mean()-(vals[m:]<=v).mean() for v in thresholds])
            j0=int(np.argmax(abs(diffs)));v=float(thresholds[j0]);h=vals<=v
            pools['structural'].append((abs(diffs[j0])/max(h.std(),1e-8),dict(kind='structural_threshold',family='structural',column=j,name=name,threshold=v)))
    for family in pools:
        descriptors.extend(d for _,d in sorted(pools[family],key=lambda a:-a[0])[:4])
    descriptors.append(dict(kind='dense',family='local',components=dense))
    return descriptors

def evaluate(descriptors,z,g):
    s,names=structural(z,g);columns=[]
    def category(desc):
        table={tuple(k):v for k,v in zip(desc['keys'],desc['scores'])}
        return np.array([table.get(tuple(row),0.) for row in z[:,desc['ids']]])
    for desc in descriptors:
        kind=desc['kind']
        if kind=='categorical':h=category(desc)
        elif kind=='threshold':h=(z[:,desc['ids'][0]]<=desc['threshold']).astype(float)
        elif kind=='structural':h=s[:,desc['column']]
        elif kind=='structural_threshold':h=(s[:,desc['column']]<=desc['threshold']).astype(float)
        elif kind=='dense':h=np.sum([category(d) for d in desc['components']],axis=0)
        else:raise ValueError(kind)
        columns.append(h)
    return np.column_stack(columns)

def prepare(x,y,g):
    cutx=len(x)//2;cuty=len(y)//2
    a,b=x[:cutx],y[:cuty];X,Y=x[cutx:],y[cuty:];z=np.r_[X,Y]
    desc=fit_features(a,b,g);F=evaluate(desc,z,g)
    families=[d['family'] for d in desc]
    # Global multiscale empirical kernel witnesses, selection half only.
    d=distance(z,np.r_[a,b])
    for gamma in SCALES:
        K=np.exp(-gamma*d);F=np.column_stack([F,K[:,:len(a)].mean(1)-K[:,len(a):].mean(1)])
        families.append('multiscale')
    F-=F.mean(0);sd=F.std(0)
    F/=np.where(sd>1e-12,sd,1.)
    # Full-vector characteristic component on evaluation rows, positive weight.
    K=np.exp(-distance(z,z));np.fill_diagonal(K,0.)
    return F,K,len(X),len(Y),families,desc

def ordered_samples(x,ids):
    if ids is None:return x
    ids=np.asarray(ids)
    if len(ids)!=len(x) or len(np.unique(ids))!=len(ids):raise ValueError('Unique ancillary observation identifiers required')
    return x[np.argsort(ids,kind='stable')]

def joint_test(x,y,g,generator,B=19999,futility=None,return_null=False,x_ids=None,y_ids=None):
    x=ordered_samples(x,x_ids);y=ordered_samples(y,y_ids)
    begin=time.monotonic();F,K,m,n,families,desc=prepare(x,y,g)
    N=m+n;scale=kernel_scale(K,m,n);fs=np.sqrt(1/m+1/n)
    def stats(A):
        v=np.abs((A@F)/m-(F.sum(0)[None,:]-A@F)/n)/fs
        u=u_stat(K,A,m,n)/scale
        subset=lambda f:np.max(v[:,np.array([k in f for k in families])],axis=1)
        local=subset(('local',));inter=subset(('interaction',));struct=subset(('structural',));multi=subset(('multiscale',))
        return np.column_stack([np.maximum(v.max(1),u),local,inter,struct,multi,u])
    obs=stats(np.r_[np.ones(m),np.zeros(n)][None,:])[0];ge=np.zeros(len(obs),int);null=[];done=0
    for start in range(0,B,128):
        A=masks(N,m,min(128,B-start),generator);values=stats(A);ge+=(values>=obs[None,:]-1e-12).sum(0);done+=len(A)
        if return_null:null.extend(values.tolist())
        if futility is not None and np.all(ge>=futility):break
    pv=(ge+1)/(B+1) if done==B else np.ones(len(obs))
    names=('omnibus','local','interaction','structural','split_multiscale','split_global')
    return dict(p=float(pv[0]),statistics=dict(zip(names,map(float,obs))),p_values=dict(zip(names,map(float,pv))),B=B,permutations_used=done,
                exceedances=ge.tolist(),p_kind='plus_one' if done==B else 'conservative_futility',selection_per_class=[len(x)//2,len(y)//2],evaluation_per_class=[m,n],
                selected=desc,seconds=time.monotonic()-begin,**(dict(null=null) if return_null else {}))

def multiscale_test(x,y,generator,B=199,futility=None,return_null=False,scales=SCALES):
    begin=time.monotonic();z=np.r_[x,y];m,n=len(x),len(y);d=distance(z,z);kernels=[];ss=[]
    for gamma in scales:
        K=np.exp(-gamma*d);np.fill_diagonal(K,0);kernels.append(K);ss.append(kernel_scale(K,m,n))
    def stats(A):
        U=np.column_stack([u_stat(K,A,m,n) for K in kernels]);Z=U/ss
        j=scales.index(1.)
        return np.column_stack([U[:,j],Z.max(1),logsumexp(Z,axis=1)-np.log(len(scales))])
    obs=stats(np.r_[np.ones(m),np.zeros(n)][None,:])[0];ge=np.zeros(3,int);done=0;null=[]
    for start in range(0,B,128):
        V=stats(masks(m+n,m,min(128,B-start),generator));ge+=(V>=obs[None,:]-1e-14*min(1,500/(m+n))).sum(0);done+=len(V)
        if return_null:null.extend(V.tolist())
        if futility is not None and np.all(ge>=futility):break
    pv=(ge+1)/(B+1) if done==B else np.ones(3)
    return dict(p_values=dict(zip(('old_hamming','multi_max','multi_softmax'),map(float,pv))),statistics=dict(zip(('old_hamming','multi_max','multi_softmax'),map(float,obs))),
                B=B,permutations_used=done,p_kind='plus_one' if done==B else 'conservative_futility',seconds=time.monotonic()-begin,**(dict(null=null) if return_null else {}))

def multinomial_test(x,y,generator,B=199):
    z=np.r_[x,y];m=len(x);_,inv=np.unique(z,axis=0,return_inverse=True);S=int(inv.max())+1
    totals=np.bincount(inv,minlength=S)
    def statistic(a):
        b=totals-a;e=totals*.5
        return 2*np.sum(np.where(a>0,a*np.log(np.maximum(a,1)/e),0)+np.where(b>0,b*np.log(np.maximum(b,1)/e),0))
    obs=statistic(np.bincount(inv[:m],minlength=S));ge=0
    for _ in range(B):ge+=statistic(np.bincount(inv[generator.choice(len(z),m,False)],minlength=S))>=obs-1e-12
    return dict(p=(ge+1)/(B+1),statistic=float(obs),support_observed=S,B=B)
