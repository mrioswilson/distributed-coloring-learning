"""Fresh shared graph cohorts, with exact declared maximum degree."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from dcl.graphs import connected_random_regular_graph
from dcl.m4_data import _edge_index
from dcl.study.graphs import degree_stats, fingerprint
from .common import atomic_json, digest


def seeds(namespace, degree, n, family, index, attempt=0):
    base = ["ucgnn-extreme-v1",namespace,degree,n,family,index,attempt]
    return int(digest(base+["graph"])[:16],16), int(digest(base+["tape"])[:16],16)


def generate(n, degree, family, seed):
    if n <= degree or n*degree % 2:
        raise ValueError("Require N>Delta and even N*Delta")
    rng = np.random.default_rng(seed)
    if family in ("REGULAR","IRREGULAR"):
        g = connected_random_regular_graph(n,degree,seed)
        if family == "IRREGULAR":
            # Protect a spanning tree AND all edges incident on a chosen hub.
            # Thus this family attains Delta, unlike the legacy <=Delta family.
            hub = int(rng.integers(n))
            protected = {tuple(sorted(e)) for e in nx.bfs_edges(g,hub)}
            protected.update(tuple(sorted((hub,v))) for v in g.neighbors(hub))
            removable = sorted(tuple(sorted(e)) for e in g.edges() if tuple(sorted(e)) not in protected)
            count = max(1,int(len(removable)*rng.uniform(.2,.85)))
            if removable:
                g.remove_edges_from(removable[i] for i in rng.permutation(len(removable))[:count])
    elif family == "HETEROGENEOUS":
        order = rng.permutation(n)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((int(order[i]),int(order[(i+1)%n])) for i in range(n))
        hubs = order[:max(1,min(4,n//100))]
        for hub in hubs:
            candidates = [int(v) for v in rng.permutation(n) if v != hub and not g.has_edge(hub,int(v))]
            candidates.sort(key=lambda v:g.degree(v))
            for v in candidates:
                if g.degree(hub) >= degree:
                    break
                if g.degree(v) < degree:
                    g.add_edge(int(hub),v)
    elif family == "BIPARTITE":
        if n%2 or degree > n//2:
            raise ValueError("Balanced regular bipartite graphs require even N>=2*Delta")
        m = n//2
        # Circulant regular bipartite stress ensemble, not a claim of uniform
        # sampling from all bipartite regular graphs. Offsets 0,1 connect it.
        offsets = [0,1] + rng.choice(np.arange(2,m),size=degree-2,replace=False).tolist()
        labels = rng.permutation(n)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((int(labels[i]),int(labels[m+(i+j)%m])) for i in range(m) for j in offsets)
    elif family == "CLIQUE":
        if n != degree+1:
            raise ValueError("A clique stress case has N=Delta+1")
        g = nx.complete_graph(n)
    else:
        raise ValueError(f"Unknown family {family}")
    assert nx.is_connected(g) and not nx.number_of_selfloops(g)
    assert max(dict(g.degree()).values()) == degree
    if family in ("REGULAR","BIPARTITE","CLIQUE"):
        assert set(dict(g.degree()).values()) == {degree}
    return g


def prepare(root, meta, historical_fingerprints, historical_seeds):
    root = Path(root)
    identity = digest({k:meta[k] for k in ("namespace","degree","n","family","graph_index")})[:24]
    info_path = root / "graphs" / f"{identity}.json"
    edges_path = info_path.with_suffix(".npz")
    if info_path.exists():
        info = json.loads(info_path.read_text())
        edges = np.load(edges_path,allow_pickle=False)["edge_index"]
        return edges,info
    used_path = root / "graph-fingerprints.json"
    used = json.loads(used_path.read_text()) if used_path.exists() else {}
    for attempt in range(50):
        graph_seed,tape_seed = seeds(meta["namespace"],meta["degree"],meta["n"],meta["family"],meta["graph_index"],attempt)
        if graph_seed in historical_seeds:
            continue
        graph = generate(meta["n"],meta["degree"],meta["family"],graph_seed)
        fp = fingerprint(graph)
        # A complete graph has only one labelled edge set at a given N.
        # Reusing it for explicit CPU/CUDA cross-checks is a deterministic
        # stress test, never a fresh independent random-graph sample.
        if fp not in historical_fingerprints and (meta["family"]=="CLIQUE" or fp not in used or used[fp] == identity):
            break
    else:
        raise RuntimeError("No fresh unique graph after 50 attempts")
    edges = _edge_index(graph)
    info = {**meta,"graph_seed":graph_seed,"tape_seed":tape_seed,"fingerprint":fp,
            "graph_id":identity,"generation_attempt":attempt,"undirected_edges":graph.number_of_edges(),
            "deterministic_stress_graph":meta["family"]=="CLIQUE",
            "reused_deterministic_stress_graph":meta["family"]=="CLIQUE" and fp in used and used[fp]!=identity,
            **degree_stats(graph),"low_degree_2_or_3_fraction":sum(d in (2,3) for _,d in graph.degree())/len(graph)}
    info_path.parent.mkdir(parents=True,exist_ok=True)
    temporary = edges_path.with_suffix(".tmp.npz")
    np.savez(temporary,edge_index=edges)
    temporary.replace(edges_path)
    atomic_json(info_path,info)
    used[fp] = identity
    atomic_json(used_path,used)
    return edges,info
