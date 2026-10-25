"""Exact maximum-degree generators, with explicit infeasible cells."""
import networkx as nx
import numpy as np
from functools import lru_cache
from dcl.multiteacher.programs import named_seed
from dcl.study.graphs import fingerprint

FAMILIES = ("regular", "random", "bipartite", "tree", "irregular", "hubs", "disconnected", "isolates", "heterogeneous", "cliques")


@lru_cache(maxsize=6000)
def graph_for(family, n, delta, seed):
    if not 1 <= delta < n:
        raise ValueError("A simple graph requires delta < n")
    rng = np.random.default_rng(seed)
    s = int(seed % (2**32 - 1))
    if family == "regular":
        if n * delta % 2:
            raise ValueError("Odd regular degree sum")
        g = nx.random_regular_graph(delta, n, seed=s)
    elif family == "bipartite":
        if delta > n // 2:
            # K_(1,delta) is still a valid bipartite exact-degree graph.
            g = nx.star_graph(delta)
            g.add_nodes_from(range(n))
        else:
            left, right = n // 2, n - n // 2
            shifts = rng.choice(right, delta, replace=False)
            g = nx.empty_graph(n)
            g.add_edges_from((v, left + int((v + shift) % right)) for v in range(left) for shift in shifts)
    elif family == "cliques":
        g = nx.empty_graph(n)
        for start in range(0, n, delta + 1):
            g.add_edges_from(nx.complete_graph(range(start, min(n, start + delta + 1))).edges())
    else:
        g = nx.star_graph(delta)
        g.add_nodes_from(range(n))
        if family == "tree":
            for v in range(delta + 1, n):
                candidates = [u for u in range(1, v) if g.degree(u) < delta]
                g.add_edge(int(rng.choice(candidates)), v)
        elif family in ("random", "irregular", "heterogeneous", "hubs"):
            # Vertex zero remains the exact-degree witness. Degree caps are
            # checked before each edge; this is not an unconditioned ER law.
            target = n * delta // (3 if family == "random" else 5)
            weights = rng.lognormal(0, 1.5, n) if family == "heterogeneous" else np.ones(n)
            if family == "hubs":
                weights[1:max(2, n // 16)] = 20
            weights[0] = 0
            weights /= weights.sum()
            for _ in range(max(10, target * 5)):
                if g.number_of_edges() >= target:
                    break
                u, v = map(int, rng.choice(n, 2, replace=False, p=weights))
                if not g.has_edge(u, v) and g.degree(u) < delta and g.degree(v) < delta:
                    g.add_edge(u, v)
        elif family == "disconnected":
            for v in range(delta + 1, n - 1, 2):
                g.add_edge(v, v + 1)
        elif family == "isolates":
            pass
        else:
            raise ValueError(f"Unknown family {family}")
    perm = list(map(int, rng.permutation(n)))
    g = nx.relabel_nodes(g, dict(enumerate(perm)))
    assert len(g) == n and max(dict(g.degree()).values()) == delta
    assert not g.is_directed() and not nx.number_of_selfloops(g)
    return g


def make_case(namespace, n, delta, family, index, replica):
    gs = named_seed(namespace, "graph", n, delta, family, index)
    g = graph_for(family, n, delta, gs)
    return g, dict(case_id=f"{namespace}-{n}-{delta}-{family}-{index}-r{replica}",
        n=n, delta=delta, actual_delta=max(dict(g.degree()).values()), family=family,
        graph_seed=gs, tape_seed=named_seed(namespace, "tape", n, delta, family, index, replica),
        fingerprint=fingerprint(g), edges=[list(e) for e in sorted(g.edges())])
