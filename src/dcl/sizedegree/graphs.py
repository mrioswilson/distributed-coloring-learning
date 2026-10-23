"""Independent labeled graphs; no graph is selected using algorithm outcomes."""
import argparse
import networkx as nx
import numpy as np
from .common import *
from dcl.globaldist.core import graph_record

def regular(n, d, rep, phase='main'):
    for attempt in range(100):
        s = seed('graph', phase, 'regular', n, d, rep, attempt)
        g = nx.random_regular_graph(d, n, seed=s)
        if nx.is_connected(g):
            row = graph_record(f'{phase}-regular-n{n}-d{d}-r{rep}', g, 'regular', phase, s)
            row.update(rep=rep, generation_attempt=attempt, generator='networkx.random_regular_graph; condition on connected')
            return enrich(row)
    raise RuntimeError('Connected regular generation failed')

def connect_components(g):
    """Degree-preserving cycle-edge switches; minimum degree is at least two."""
    while not nx.is_connected(g):
        parts = sorted(nx.connected_components(g), key=lambda s: min(s))
        cycles = [nx.cycle_basis(g.subgraph(p))[0] for p in parts[:2]]
        (a,b),(c,d) = [(v[0],v[1]) for v in cycles]
        g.remove_edges_from([(a,b),(c,d)])
        g.add_edges_from([(a,c),(b,d)])
    return g

def irregular(n, d, rep=0):
    # Three target degrees keep receiver-local palette heterogeneity explicit.
    s = seed('graph', 'secondary', 'irregular', n, d, rep)
    r = np.random.default_rng(s)
    nh = max(1, n//100)
    nm = n*35//100
    middle = max(2, d//3)
    degree = np.array([d]*nh + [middle]*nm + [2]*(n-nh-nm))
    if degree.sum() % 2:
        # Change one middle target by one, prospectively and reproducibly.
        degree[nh] += 1 if middle < d else -1
    degree = degree[r.permutation(n)]
    if not nx.is_graphical(degree.tolist()):
        raise RuntimeError(f'Non-graphical frozen heterogeneous sequence {n}, {d}')
    g = connect_components(nx.havel_hakimi_graph(degree.tolist()))
    performed = nx.connected_double_edge_swap(g, nswap=10*g.number_of_edges(), seed=s)
    row = graph_record(f'secondary-irregular-n{n}-d{d}-r{rep}', g, 'irregular', 'secondary', s)
    row.update(rep=rep, generator='shuffled Havel-Hakimi; cycle switches to connect; connected double-edge swaps', swaps=performed)
    return enrich(row)

def enrich(row):
    d = np.array(row['degrees'])
    vals, counts = np.unique(d, return_counts=True)
    row.update(degree_variance=float(d.var()), near_max_fraction=float((d >= .9*row['delta']).mean()),
               degree_histogram={str(v): int(c) for v,c in zip(vals,counts)},
               palette_histogram={str(v+1): int(c) for v,c in zip(vals,counts)},
               rho=row['n']/(row['delta']+1), density=float(d.mean()/(row['n']-1)))
    return row

def validate(rows):
    hashes=set()
    for row in rows:
        g=nx.Graph();g.add_nodes_from(range(row['n']))
        e=np.asarray(row['edges']).reshape(2,-1)
        g.add_edges_from(e.T.tolist())
        assert not list(nx.selfloop_edges(g))
        assert nx.is_connected(g)
        assert sorted(g)==list(range(row['n']))
        assert [g.degree(i) for i in range(row['n'])]==row['degrees']
        assert e.shape[1]==2*g.number_of_edges()
        assert len(set(map(tuple,e.T)))==e.shape[1]
        assert max(row['degrees'])==row['delta']
        if row['family']=='regular': assert set(row['degrees'])=={row['delta']}
        assert row['sha256'] not in hashes
        hashes.add(row['sha256'])
        assert graph_record(row['id'],g,row['family'])['sha256']==row['sha256']
    return dict(passed=True,graphs=len(rows),unique_labeled_hashes=len(hashes))

def main():
    p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=3);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'freeze.json').exists(): raise RuntimeError('Already frozen')
    pilot=[regular(n,d,0,'pilot') for n in NS for d in DS]
    primary=[regular(n,d,r) for r in range(args.reps) for n in NS for d in DS]
    secondary=[irregular(n,d) for n in NS for d in DS]
    old=next(g for g in read(GLOBAL/'results/ucgnn-global-distribution/graphs.json') if g['id']=='regular-n512-d100')
    historical=enrich(dict(old,id='historical-regular-n512-d100',stage='historical-replication',rep=0,original_id=old['id']))
    rows=primary+secondary+[historical]
    atomic_json(OUT/'graphs.json',rows)
    atomic_json(OUT/'candidate-graphs.json',rows)
    atomic_json(OUT/'pilot-graphs.json',pilot)
    atomic_json(OUT/'graph-audit.json',validate(rows+pilot))
    atomic_json(OUT/'checkpoints.json',read(BASE/'results/ucgnn-robust-distribution/checkpoints.json'))
    print(json.dumps(dict(primary=len(primary),secondary=len(secondary),historical=1,pilot=len(pilot))))

if __name__=='__main__':main()
