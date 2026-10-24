"""Independent splits, fixed horizons, frozen source/checkpoint identities."""
import json
from pathlib import Path

import networkx as nx
import numpy as np

from dcl.multiteacher.programs import named_seed
from dcl.samplecomplexity.protocol import graph_for, horizon, summarize_runs, five_seed_gate, freeze, passed_run
from dcl.study.graphs import fingerprint
from dcl.study.storage import file_hash

FAMILIES = ("path", "cycle", "cubic", "irregular", "bipartite", "symmetric", "adversarial",
            "edgeless", "mixed", "cliques-and-isolates")


def cases(config, split, seed):
    if split not in ("pilot", "development", "final", "transfer"):
        raise ValueError("Evaluation split required")
    count, sizes = config[f"{split}_runs"], config[f"{split}_sizes"]
    result = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        n = sizes[(i // len(FAMILIES)) % len(sizes)]
        gs = named_seed(config["namespace"], split, seed, "graph", i)
        if family == "edgeless":
            g = nx.empty_graph(n)
        elif family in ("mixed", "cliques-and-isolates"):
            rng = np.random.default_rng(gs)
            isolated = int(rng.integers(max(2, n // 8), n // 3 + 1))
            if family == "mixed":
                core = nx.path_graph(n - isolated)
            else:
                core = nx.disjoint_union_all([nx.complete_graph(4) for _ in range((n - isolated) // 4)])
            g = nx.disjoint_union(core, nx.empty_graph(n - len(core)))
            g = nx.relabel_nodes(g, dict(enumerate(map(int, rng.permutation(n)))))
        else:
            g = graph_for(family, n, gs)
        meta = dict(case_id=f"{split}-s{seed}-{i:04d}", seed=seed, family=family, n=n,
            graph_seed=gs, fingerprint=fingerprint(g), edges=[list(e) for e in sorted(g.edges())],
            tape_seed=named_seed(config["namespace"], split, seed, "tape", i))
        result.append((g, meta))
    return result


def source_hashes():
    paths = [*Path("src/dcl/multicf").glob("*.py"), *Path("src/dcl/multiteacher").glob("*.py"),
        *Path("src/dcl/multiteacher_e1").glob("*.py"), Path("src/dcl/samplecomplexity/model.py"),
        Path("src/dcl/samplecomplexity/training.py"), Path("src/dcl/samplecomplexity/evaluation.py"),
        Path("src/dcl/integration/models.py"), Path("src/dcl/paperstudy/models.py"),
        *Path("src/dcl/universal").glob("*.py")]
    return {str(p): file_hash(p) for p in sorted(paths)}
