from __future__ import annotations
import hashlib, json, itertools, time
from pathlib import Path
from functools import lru_cache
from fractions import Fraction
import numpy as np
import networkx as nx
from dcl.localpalettes.cohort import edge_array
from dcl.localpalettes.config import model_specs
from dcl.study.storage import atomic_json, file_hash

ROOT=Path('data/ucgnn-global-distribution')
OUT=Path('results/ucgnn-global-distribution')

def seed(*args):
    return int.from_bytes(hashlib.sha256(json.dumps(['global-law-v1',*args]).encode()).digest()[:8],'big')%(2**63)

def specs():
    return [s for s in model_specs() if s['treatment'] in ('normalized','deepsets','transformer')]

def graph_record(name,g,family,stage='confirmatory',graph_seed=None):
    e=edge_array(g); d=np.array([g.degree(i) for i in range(len(g))],dtype=np.int64)
    return dict(id=name,n=len(g),delta=int(d.max()),family=family,stage=stage,graph_seed=graph_seed,
                edges=e.tolist(),degrees=d.tolist(),sha256=hashlib.sha256(np.int64(len(g)).tobytes()+e.tobytes()).hexdigest(),
                mean_degree=float(d.mean()),degree_cv=float(d.std()/max(d.mean(),1e-9)))

def small_graphs():
    return [graph_record(name,g,f,'small') for name,g,f in [
        ('edge',nx.path_graph(2),'path'),('path3',nx.path_graph(3),'path'),
        ('path4',nx.path_graph(4),'path'),('triangle',nx.complete_graph(3),'dense'),
        ('cycle4',nx.cycle_graph(4),'cycle'),('star3',nx.star_graph(3),'star'),
        ('paw',nx.Graph([(0,1),(1,2),(2,0),(2,3)]),'irregular')]]

def graphs():
    rows=small_graphs()
    # Two fixed sizes separate n and maximum degree, with identical generation rules.
    for n in (256,512):
        for d in (2,3,5,10,20,50,100,200):
            s=seed('graph','regular',n,d)
            rows.append(graph_record(f'regular-n{n}-d{d}',nx.random_regular_graph(d,n,seed=s),'regular',graph_seed=s))
    for d in (2,3,5,10,20,50,100,200):
        rows.append(graph_record(f'star-d{d}',nx.star_graph(d),'star'))
    rows += [graph_record('path256',nx.path_graph(256),'path'),graph_record('cycle256',nx.cycle_graph(256),'cycle')]
    for d in (3,20,100,200):
        n=256;s=seed('graph','random',n,d);rng=np.random.default_rng(s)
        g=nx.gnp_random_graph(n,.7*d/(n-1),seed=s)
        # Force actual maximum d while retaining all vertices: hub plus capped remainder.
        g.remove_edges_from(list(g.edges(0)));g.add_edges_from((0,v) for v in range(1,d+1))
        for v in range(1,n):
            while g.degree(v)>d:
                w=next(w for w in g[v] if w!=0);g.remove_edge(v,w)
        rows.append(graph_record(f'random-n256-d{d}',g,'random',graph_seed=s))
        g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from((0,v) for v in range(128,128+min(d,128)))
        # d=200 uses a 56/200 bipartition, other cells use 128/128.
        a=56 if d==200 else 128;b=n-a;g=nx.Graph();g.add_nodes_from(range(n))
        g.add_edges_from((0,v) for v in range(a,a+d))
        for u in range(1,a):
            for v in rng.choice(np.arange(a,n),size=max(1,min(d//3,b)),replace=False):g.add_edge(u,int(v))
        rows.append(graph_record(f'bipartite-n256-d{d}',g,'bipartite',graph_seed=s))
        g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from((0,v) for v in range(1,d+1))
        # Connected bounded-degree tree, exactly prescribed hub degree.
        for v in range(d+1,n):g.add_edge(max(1,v-d),v)
        rows.append(graph_record(f'tree-n256-d{d}',g,'tree'))
    for d in (3,20,100,200):
        rows.append(graph_record(f'wheel-d{d}',nx.wheel_graph(d+1),'hub'))
        rows.append(graph_record(f'clique-d{d}',nx.complete_graph(d+1),'dense'))
    return rows

def valid_colors(graph):
    edges=np.array(graph['edges'],dtype=np.int64); degrees=graph['degrees']
    return np.array([x for x in itertools.product(*(range(d+1) for d in degrees))
                     if all(x[u]!=x[v] for u,v in edges.T)],dtype=np.int16)

def exact_teacher(graph):
    """Rational absorbing law, eliminating self loops by monotone fixed-color count."""
    n=graph['n'];adj=[set() for _ in range(n)]
    for u,v in np.array(graph['edges']).T:adj[u].add(v)
    @lru_cache(None)
    def solve(state):
        pending=[v for v,c in enumerate(state) if c<0]
        if not pending:return {state:Fraction(1)}
        choices=[]
        for v in pending:
            a=[c for c in range(graph['degrees'][v]+1) if all(state[w]!=c for w in adj[v])]
            choices.append([(-1,Fraction(1,2))]+[(c,Fraction(1,2*len(a))) for c in a])
        trans={}
        for offers in itertools.product(*choices):
            cand=dict(zip(pending,[x[0] for x in offers]));q=Fraction(1)
            for _,p in offers:q*=p
            new=list(state)
            for v,c in cand.items():
                if c>=0 and all(cand.get(w,-1)!=c for w in adj[v]):new[v]=c
            t=tuple(new);trans[t]=trans.get(t,Fraction(0))+q
        stay=trans.pop(state,Fraction(0));out={}
        for t,p in trans.items():
            for x,q in solve(t).items():out[x]=out.get(x,Fraction(0))+p*q/(1-stay)
        assert sum(out.values())==1
        return out
    law=solve(tuple([-1]*n));support=valid_colors(graph)
    return support,np.array([float(law.get(tuple(x),0)) for x in support]),{','.join(map(str,x)):str(p) for x,p in law.items()},solve.cache_info().currsize


def kernel(x,y,gamma=1.):
    # Bound intermediate memory; the observation remains the entire row.
    return np.concatenate([np.exp(-gamma*np.mean(a[:,None,:]!=y[None,:,:],axis=2)) for a in np.array_split(x,max(1,(len(x)+63)//64))])

def mmd_test(x,y,rng,B=19999,gamma=1.):
    return mmd_from_kernel(kernel(np.concatenate((x,y)),np.concatenate((x,y)),gamma),len(x),len(y),rng,B,gamma)

def mmd_from_kernel(K,m,n,rng,B=19999,gamma=1.):
    if min(m,n)<2:return dict(error='insufficient successful observations')
    K=K.copy();np.fill_diagonal(K,0)
    # U statistic. Fixed sample sizes permit reusing cross row sums.
    def stat(A):
        AK=A@K;aa=np.sum(AK*A,axis=1);ar=AK.sum(1);tot=K.sum()
        bb=tot-2*ar+aa;ab=ar-aa
        return aa/(m*(m-1))+bb/(n*(n-1))-2*ab/(m*n)
    a=np.r_[np.ones(m),np.zeros(n)][None,:];obs=float(stat(a)[0]);ge=0
    tie_tolerance=1e-14*min(1.,500/(m+n))
    for start in range(0,B,256):
        b=min(256,B-start);A=np.zeros((b,m+n))
        for r in range(b):A[r,rng.choice(m+n,m,replace=False)]=1
        ge+=int(np.sum(stat(A)>=obs-tie_tolerance))
    # Distribution-free bounded-difference confidence interval for biased MMD norm.
    Kfull=K.copy();np.fill_diagonal(Kfull,1)
    w=np.r_[np.full(m,1/m),np.full(n,-1/n)]
    norm=float(np.sqrt(max(0,w@Kfull@w)))
    radius=(1/np.sqrt(m)+1/np.sqrt(n))*(1+np.sqrt(2*np.log(2/.05)))
    return dict(m=m,n=n,mmd2_unbiased=obs,mmd_biased=norm,mmd_population_ci95=[max(0,norm-radius),min(np.sqrt(2),norm+radius)],
                p=(ge+1)/(B+1),B=B,gamma=gamma,permutation_tie_tolerance=tie_tolerance,ci_method='distribution-free RKHS mean concentration; conservative')

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));v=0
    for i,j in enumerate(order):v=max(v,(len(p)-i)*p[j]);out[j]=min(1,v)
    return out


def tv_interval(counts,p,alpha=.05):
    q=counts/counts.sum();tv=float(np.abs(q-p).sum()/2);k=len(p);m=int(counts.sum())
    # Bretagnolle-Huber-Carol/union-over-subsets bound, proved in theory document.
    radius=np.sqrt((k*np.log(2)+np.log(1/alpha))/(2*m))
    return tv,[max(0,tv-radius),min(1,tv+radius)]
