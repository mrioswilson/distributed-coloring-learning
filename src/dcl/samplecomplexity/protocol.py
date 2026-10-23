"""Frozen design, independent graph/tape namespaces, and strict certification."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from dcl.multiteacher.programs import TEACHERS, named_seed
from dcl.study.graphs import fingerprint
from dcl.study.storage import atomic_json, file_hash

GRID = (1000, 3000, 10000, 30000, 100000, 300000, 1000000)
FAMILIES = ("path", "cycle", "cubic", "irregular", "bipartite", "symmetric", "adversarial")
REGIME = {t: ("A" if t in ("LUBY", "JOHANSSON", "COLOR_TRIALS") else "B") for t in TEACHERS}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def horizon(teacher, n):
    """Native steps. Randomized deadlines bound ideal per-run tail by 1e-12."""
    rho = {"LUBY": .25, "JOHANSSON": .125, "COLOR_TRIALS": 27 / 256}
    if teacher in rho:
        return math.ceil(math.log(1e-12 / n) / math.log1p(-rho[teacher]))
    return {"JP": n, "GREEDY": n, "BE_BASE_D3": 15,
            "SPECULATIVE_GM": 2 * math.ceil(n / min(4, n)) + n}[teacher]


def graph_for(family, n, seed):
    if family == "path":
        g = nx.path_graph(n)
    elif family == "cycle":
        g = nx.cycle_graph(n)
    elif family == "symmetric":
        g = nx.circular_ladder_graph(n // 2)
    elif family == "adversarial":
        # Disconnected K4 components: all four colors necessary; valid native input.
        g = nx.disjoint_union_all([nx.complete_graph(4) for _ in range(n // 4)])
    elif family == "bipartite":
        rng = np.random.default_rng(seed)
        g = nx.cycle_graph(n)
        # Add a matching across bipartition; never exceed degree three.
        left, right = rng.permutation(np.arange(0, n, 2)), rng.permutation(np.arange(1, n, 2))
        for u, v in zip(left, right):
            g.add_edge(int(u), int(v))
    else:
        g = nx.random_regular_graph(3, n, seed=int(seed))
        if family == "irregular":
            rng = np.random.default_rng(seed)
            edges = sorted(g.edges())
            remove = rng.choice(len(edges), max(1, len(edges) // 4), replace=False)
            g.remove_edges_from(edges[i] for i in remove)
        elif family != "cubic":
            raise ValueError(family)
    # Independently relabel also the deterministic families.
    order = np.random.default_rng(named_seed(seed, "labels")).permutation(n)
    return nx.relabel_nodes(g, dict(enumerate(map(int, order))))


def cases(config, split, seed):
    if split not in ("pilot", "development", "final"):
        raise ValueError("No training examples may be drawn from evaluation cases")
    count = config[f"{split}_runs"]
    sizes = config[f"{split}_sizes"]
    result = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        n = sizes[(i // len(FAMILIES)) % len(sizes)]
        graph_seed = named_seed(config["namespace"], split, seed, "graph", i)
        g = graph_for(family, n, graph_seed)
        meta = dict(case_id=f"{split}-s{seed}-{i:04d}", seed=seed, family=family, n=n,
                    graph_seed=graph_seed, fingerprint=fingerprint(g),
                    edges=[list(e) for e in sorted(g.edges())],
                    tape_seed=named_seed(config["namespace"], split, seed, "tape", i))
        result.append((g, meta))
    return result


def passed_run(row):
    required = ("proper", "palette", "invariants", "terminated", "no_technical_error",
                "no_cutoff", "terminal_correct")
    return all(row.get(key) is True for key in required)


def summarize_runs(rows, expected_ids):
    ids = [r["case_id"] for r in rows]
    complete = len(ids) == len(set(ids)) and set(ids) == set(expected_ids)
    successes = sum(passed_run(r) for r in rows)
    ok = complete and bool(rows) and successes == len(expected_ids)
    return dict(pass_all=ok, complete=complete, executions=len(rows), successes=successes,
                failures=dict(Counter(r.get("failure") or "unspecified" for r in rows if not passed_run(r))),
                success_rate=successes / len(expected_ids) if expected_ids else 0.,
                lower_one_sided_95=(.05 ** (1 / len(rows)) if ok else None),
                mean_native_steps=float(np.mean([r["native_steps"] for r in rows])) if rows else None)


def five_seed_gate(summaries, seeds, minimum=1000):
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five independent training seeds are required")
    ok = [s for s in seeds if s in summaries and summaries[s].get("pass_all") is True
          and summaries[s].get("executions", 0) >= minimum]
    return dict(status="PASS" if len(ok) == 5 else "NOT CERTIFIED", seeds_pass=len(ok),
                seeds_required=5, missing_seeds=[s for s in seeds if s not in summaries])


def source_hashes():
    # Hash executable dependencies, excluding manuscript/result changes.
    paths = [*[Path("src/dcl/samplecomplexity")/name for name in
               ("protocol.py", "model.py", "data.py", "evaluation.py", "interactive.py", "training.py", "runner.py", "controls.py")],
             *Path("src/dcl/multiteacher").glob("*.py"),
             *Path("src/dcl/multiteacher_e1").glob("*.py"),
             Path("src/dcl/integration/models.py"), Path("src/dcl/paperstudy/models.py"),
             *Path("src/dcl/universal").glob("*.py"), Path("src/dcl/study/graphs.py"),
             Path("src/dcl/study/storage.py"), Path("src/dcl/graphs.py")]
    return {str(p): file_hash(p) for p in sorted(paths)}


def freeze(path, payload):
    """Refuse accidental replacement of preregistration or consumed evidence."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"Frozen object differs: {path}")
    else:
        atomic_json(path, payload)
