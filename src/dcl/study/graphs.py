import hashlib
import networkx as nx
import numpy as np

from dcl.graphs import connected_random_regular_graph


def fingerprint(graph):
    edges = np.asarray(sorted((min(u, v), max(u, v)) for u, v in graph.edges()), dtype="<i8").reshape(-1, 2)
    return hashlib.sha256(str(len(graph)).encode() + edges.tobytes()).hexdigest()


def degree_stats(graph):
    degrees = np.asarray([d for _, d in graph.degree()])
    return {"degree_min": int(degrees.min()), "degree_max": int(degrees.max()), "degree_mean": float(degrees.mean())}


def generate_graph(n, degree, family, seed):
    family = family.upper()
    rng = np.random.default_rng(seed)
    if family == "MIXED":
        family = "REGULAR" if rng.random() < 0.5 else "IRREGULAR"
    if family == "PATH":
        graph = nx.path_graph(n)
    elif family == "CYCLE":
        graph = nx.cycle_graph(n)
    elif family in ("REGULAR", "IRREGULAR"):
        graph = connected_random_regular_graph(n, degree, seed)
        if family == "IRREGULAR":
            # Randomized DFS tree guarantees connectivity. Never remove its edges.
            root = int(rng.integers(n))
            visited, tree, stack = {root}, set(), [root]
            while stack:
                u = stack.pop()
                for v in rng.permutation(sorted(graph.neighbors(u))):
                    v = int(v)
                    if v not in visited:
                        visited.add(v)
                        tree.add(tuple(sorted((u, v))))
                        stack.append(v)
            removable = sorted(tuple(sorted(edge)) for edge in graph.edges() if tuple(sorted(edge)) not in tree)
            # Random density, including very sparse connected graphs.
            count = max(1, int(round(rng.uniform(0.1, 1.0) * len(removable))))
            order = rng.permutation(len(removable))
            graph.remove_edges_from(removable[j] for j in order[:count])
            # Rare symmetric deletions can leave a regular graph; remove one
            # further non-tree edge. A tree at these sizes cannot be regular.
            if len(set(dict(graph.degree()).values())) == 1:
                graph.remove_edge(*removable[order[count]])
    else:
        raise ValueError(f"Unknown graph distribution {family}")
    if set(graph) != set(range(n)) or not nx.is_connected(graph):
        raise RuntimeError("Generated graph must be connected with contiguous labels")
    if max(dict(graph.degree()).values()) > degree:
        raise ValueError("Graph exceeds declared maximum degree")
    graph.graph["family"] = family
    return graph
