"""Known laws on proper local-list star colorings; no neural training."""
import numpy as np
from dcl.globaldist.core import graph_record
import networkx as nx

NULLS=('point_null','uniform_null','discrete_null','star_null','clique_null')
ALTERNATIVES=('localized','localized_4','marginal','correlation','higher_order','structural','diffuse','rare','large_support','parity_global')
CASES=NULLS+ALTERNATIVES

def draw(case,m,r,side=0,D=200,effect=None):
    if case=='clique_null':
        import itertools
        g=graph_record('control-clique4',nx.complete_graph(4),'dense')
        support=np.array(list(itertools.permutations(range(4))),np.int16)
        return support[r.integers(0,len(support),m)],g,dict(tv=0.,kind='known equality by clique symmetry; simulated uniform law')
    if case in ('point_null','uniform_null','discrete_null'):
        g=graph_record('control-path3',nx.path_graph(3),'path')
        support=np.array([(0,1,0),(1,0,1),(0,2,0),(1,2,1),(0,2,1),(1,2,0)],np.int16)
        p=np.ones(6)/6
        if case=='point_null':p=np.array([1.,0,0,0,0,0])
        if case=='discrete_null':p=np.array([.1,.2,.05,.4,.15,.1])
        return support[r.choice(6,m,p=p)],g,dict(tv=0.,kind='known equality')
    g=graph_record(f'control-star-{D}',nx.star_graph(D),'star')
    z=r.integers(0,2,(m,D+1),dtype=np.int16);z[:,0]=2
    meta=dict(tv=None,kind='known difference',support_note=f'up to 2^{D} leaf vectors; no repeats needed')
    if case=='star_null':z[:,0]=r.integers(2,D+1,m);meta.update(tv=0.,kind='known equality')
    elif case=='localized':
        z[:,0]=r.integers(2,(D//2+2 if side else D+1),m)
        meta.update(tv=1-(D//2)/(D-1),known_event=f'center <= {D//2+1}')
    elif case=='localized_4':
        if side:z[:,1:5]=(r.random((m,4))<.8)
        p=np.array([.8**k*.2**(4-k) for k in range(5)]);q=np.repeat(1/16,5)
        from math import comb
        meta['tv']=.5*sum(comb(4,k)*abs(p[k]-q[k]) for k in range(5))
    elif case in ('marginal','large_support'):
        p=.85 if effect is None else effect
        if side:z[:,1]=(r.random(m)<p)
        meta['tv']=abs(p-.5)
    elif case=='correlation':
        if side:z[:,2]=z[:,1]
        meta.update(tv=.5,same_vertex_marginals=True)
    elif case=='higher_order':
        if side:z[:,3]=z[:,1]^z[:,2]
        meta.update(tv=.5,same_vertex_and_pair_marginals=True)
    elif case=='parity_global':
        if side:z[:,-1]=np.bitwise_xor.reduce(z[:,1:-1],axis=1)
        meta.update(tv=.5,same_all_proper_subset_marginals=True)
    elif case=='structural':
        if side:z[:,1:]=r.integers(0,2,(m,1))
        meta.update(tv=1-2.**(1-D),same_vertex_marginals=True)
    elif case=='diffuse':
        p=.55 if effect is None else effect
        if side:z[:,1:]=(r.random((m,D))<p)
        from scipy.stats import binom
        meta['tv']=float(.5*np.abs(binom.pmf(np.arange(D+1),D,p)-binom.pmf(np.arange(D+1),D,.5)).sum())
    elif case=='rare':
        p=.001 if effect is None else effect
        if side:z[r.random(m)<p,0]=3
        meta.update(tv=p,rare_probability=p)
        meta['power_upper_bound']=1-(1-p)**m+.05*(1-p)**m
    else:raise ValueError(case)
    return z,g,meta
