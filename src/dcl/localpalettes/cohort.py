from __future__ import annotations

import hashlib
import json

import networkx as nx
import numpy as np

from dcl.study.storage import atomic_json
from .config import RESULTS, ROOT, seed

FAMILIES = ('REGULAR','IRREGULAR','HETEROGENEOUS','STAR','BIPARTITE','HUB','DENSE','LOW_HIGH')


def edge_array(graph):
    pairs = sorted((min(a,b),max(a,b)) for a,b in graph.edges())
    return np.array(sorted(pairs+[(b,a) for a,b in pairs]),dtype=np.int64).reshape(-1,2).T.copy()


def capped(graph, d, rng):
    graph = graph.copy()
    # Remove, never silently discard vertices; isolated K=1 is meaningful.
    for v in rng.permutation(len(graph)):
        neighbors = list(graph[v])
        if len(neighbors)>d:
            rng.shuffle(neighbors)
            graph.remove_edges_from((v,w) for w in neighbors[d:])
    return graph


def make_graph(family, d, replicate, n=None, pilot=False):
    rng_seed = seed('pilot' if pilot else 'confirmatory','graph',family,d,replicate,n)
    rng = np.random.default_rng(rng_seed)
    n = n or max(80,2*d+2)
    nxseed = int(rng_seed % 2**32)
    if family == 'REGULAR':
        if n*d%2:n+=1
        g=nx.random_regular_graph(d,n,seed=nxseed)
    elif family == 'IRREGULAR':
        g=capped(nx.fast_gnp_random_graph(n,.65*d/(n-1),seed=nxseed),d,rng)
    elif family == 'HETEROGENEOUS':
        g=capped(nx.barabasi_albert_graph(n,max(1,d//8),seed=nxseed),d,rng)
    elif family == 'STAR':
        g=nx.star_graph(d)
        if replicate:g=nx.disjoint_union(g,nx.star_graph(max(1,d//2)))
    elif family == 'BIPARTITE':
        a=max(1,d//4)
        g=nx.complete_bipartite_graph(a,d)
        if replicate:
            for e in list(g.edges()):
                if e[0]!=0 and rng.random()<.35:g.remove_edge(*e)
    elif family == 'HUB':
        g=nx.wheel_graph(d+1)
        if replicate:
            g.add_nodes_from(range(d+1,2*d+2))
            g.add_edges_from((v,v+1) for v in range(d+1,2*d+1))
    elif family == 'DENSE':
        g=nx.complete_graph(d+1) if not replicate else capped(nx.gnp_random_graph(d+5,.75,seed=nxseed),d,rng)
    elif family == 'LOW_HIGH':
        g=nx.empty_graph(n)
        g.add_edges_from((0,w) for w in range(1,min(n,d+1)))
        g.add_edges_from((v,v+1) for v in range(d+1,n-1))
        if replicate:
            for v in range(d+1,n):
                if rng.random()<.5 and g.degree(v)<d and g.degree(1)<d:g.add_edge(v,1)
    else:raise ValueError(family)
    return g,rng_seed


def build_graphs():
    path=RESULTS/'graph-manifest.json'
    if path.exists():return json.loads(path.read_text())
    specs=[(f,d,r,None) for f in FAMILIES for d in (3,10,50,200) for r in (0,1)]
    specs += [(f,d,r,640) for f in ('IRREGULAR','HETEROGENEOUS','LOW_HIGH') for d in (10,50) for r in (0,1)]
    rows=[]
    for family,d,r,n in specs:
        g,s=make_graph(family,d,r,n)
        edges=edge_array(g)
        ident=f'{family.lower()}-d{d}-r{r}-n{len(g)}'
        file=RESULTS/'graphs'/f'{ident}.npz'
        file.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(file,edges=edges,degree=np.array([g.degree(i) for i in range(len(g))]))
        degrees=np.array([g.degree(i) for i in range(len(g))])
        rows.append(dict(id=ident,family=family,target_degree=d,replicate=r,n=len(g),degree=int(degrees.max()),
                         mean_degree=float(degrees.mean()),degree_cv=float(degrees.std()/max(degrees.mean(),1e-12)),
                         seed=s,edges_sha256=hashlib.sha256(edges.tobytes()).hexdigest(),file=str(file)))
    atomic_json(path,rows)
    return rows


def palette_for(regime,degrees):
    if regime=='LOCAL':return np.asarray(degrees)+1
    maximum=int(max(degrees))
    factor={'GLOBAL':1.,'OVERSIZED-1.5':1.5,'OVERSIZED-2':2.}[regime]
    return np.full(len(degrees),int(np.ceil(factor*maximum))+1,dtype=np.int64)


def state_specs():
    rows=[]
    for k in (4,11,51,101,201,501):
        patterns=[('ALL_FREE',np.arange(k))]
        patterns += [(f'SINGLE-{name}',[pos]) for name,pos in zip(('first','quarter','middle','threequarter','last'),(0,k//4,k//2,3*k//4,k-1))]
        patterns += [('TWO-adjacent-start',[0,1]),('TWO-endpoints',[0,k-1]),('TWO-middle',[k//2-1,k//2]),
                     ('THREE-spread',[0,k//2,k-1])]
        length=max(1,k//4)
        patterns += [(f'BLOCK-{name}',range(start,start+length)) for name,start in (('first',0),('middle',(k-length)//2),('last',k-length))]
        patterns += [('ALTERNATING',range(0,k,2)),('SCATTERED',np.random.default_rng(seed('state',k)).choice(k,min(7,k),replace=False))]
        seen=set()
        for name,a in patterns:
            a=tuple(sorted(set(map(int,a))))
            if a in seen:continue
            seen.add(a)
            rows.append(dict(id=f'k{k}-{name}',k=k,available=list(a),pattern=name,kind='realizable-local-neighborhood'))
    rows += [dict(id='k11-EXAMPLE',k=11,available=[1,6,10],pattern='EXAMPLE',kind='realizable-local-neighborhood'),
             dict(id='k64-EXAMPLE',k=64,available=[0,7,19,63],pattern='EXAMPLE',kind='realizable-local-neighborhood')]
    for row in rows:
        k,a=row['k'],row['available']
        forbidden=[c for c in range(k) if c not in a]
        colors=[-1]+forbidden+[-1]*(len(a)-1)
        row.update(colors=colors,palettes=[k]+[max(2,c+1) for c in forbidden]+[2]*(len(a)-1),
                   witness_private_pending_leaves=[max(0,c-1) for c in forbidden],
                   center_degree=k-1,
                   b3=(row['pattern']=='ALL_FREE' and k in (4,51,201)) or
                      (k==51 and row['pattern'] in ('SINGLE-middle','THREE-spread')) or
                      (k==201 and row['pattern']=='ALTERNATING'))
    return rows


def star_edges(n):
    leaves=np.arange(1,n,dtype=np.int64)
    return np.stack((np.r_[leaves,np.zeros(n-1,dtype=np.int64)],np.r_[np.zeros(n-1,dtype=np.int64),leaves]))


def build_states():
    path=RESULTS/'state-manifest.json'
    if not path.exists():atomic_json(path,state_specs())
    return json.loads(path.read_text())
