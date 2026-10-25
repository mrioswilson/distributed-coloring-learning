#!/usr/bin/env python3
"""Read-only COMMIT interventions and CT preference-gap mechanism probes.

These are teacher-state diagnostic queries, never repaired student rollouts.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import json
import networkx as nx
import numpy as np
import torch
from dcl.degree.inventory import ROOT
from dcl.degree.adapter import FrozenAdapter
from dcl.degree.evaluation import component_diagnostics
from dcl.degree.graphs import make_case
from dcl.degree.teachers import DegreeProgram, exact_choices
from dcl.multiteacher.programs import DETECT, named_seed
from dcl.samplecomplexity.model import QueryPolicy
from dcl.study.storage import atomic_json, file_hash


def task(cp):
    torch.set_num_threads(1)
    model = QueryPolicy(cp["teacher"], cp["architecture"])
    model.load_state_dict(torch.load(cp["path"], weights_only=False)["model"])
    adapter = FrozenAdapter(model)
    counts = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    witnesses = []
    replica = cp["seed"] % 100 - 1
    for delta in (3, 4, 10, 50, 200):
        for palette in ("global", "local"):
            for family in ("isolates", "bipartite"):
                n = 32 if delta < 32 else 256
                # Sparse star for high-degree diagnosis bounds cost, with a
                # distinct graph family explicitly recorded in the probe.
                f = family if delta <= 10 else "isolates"
                g, meta = make_case("degree-cf-probe-20261025", n, delta, f, int(family == "bipartite"), replica)
                p = DegreeProgram(g, cp["teacher"], meta["tape_seed"], palette)
                sampled = 0
                phase_samples = defaultdict(int)
                while not p.finished and (cp["teacher"] == "SPECULATIVE_GM" or sampled < 2):
                    state = p.snapshot()
                    if state.phase == DETECT and phase_samples[state.phase] < 2:
                        d = adapter(p, state)
                        truth = exact_choices(p, state).mark[d.roots]
                        key = f"d{delta}-{palette}-detection"
                        a = counts[key]
                        a[0] += int((d.commit == truth).sum()); a[1] += len(truth)
                        a[2] += int((d.commit & ~truth).sum()); a[3] += int((~d.commit & truth).sum())
                        a[4] += int(truth.sum()); a[5] += int((~truth).sum())
                        phase_samples[state.phase] += 1
                    elif state.phase != DETECT and phase_samples[state.phase] < 2:
                        rng = np.random.default_rng(named_seed("cf-probe", replica, delta, palette, family, p.steps))
                        for variant in range(4):
                            participation = state.permission.copy()
                            candidates = (rng.random(p.n) * p.palette).astype(np.int64)
                            if variant == 1:
                                candidates[:] = 0  # common typed same-color offers
                            if variant == 2:
                                participation &= rng.random(p.n) < .5
                            if variant == 3:
                                participation[:] = False
                            d = adapter(p, state, overrides=dict(participation=participation, candidates=candidates,
                                                                 query=candidates))
                            predicted, target = component_diagnostics(p, state, d)["commit_matched"]
                            key = f"d{delta}-{palette}"
                            a = counts[key]
                            a[0] += int((predicted == target).sum()); a[1] += len(target)
                            a[2] += int((predicted & ~target).sum()); a[3] += int((~predicted & target).sum())
                            a[4] += int(target.sum()); a[5] += int((~target).sum())
                            if len(witnesses) < 2 and np.any(predicted != target):
                                idx = int(np.flatnonzero(predicted != target)[0])
                                witnesses.append(dict(delta=delta, palette=palette, family=f, case=meta,
                                    step=p.steps, variant=variant, root=int(d.roots[idx]),
                                    query=int(d.candidates[idx]), participation=bool(d.participation[idx]),
                                    logit=float(d.commit_logits[idx]), predicted=bool(predicted[idx]), expected=bool(target[idx])))
                        sampled += 1
                        phase_samples[state.phase] += 1
                    p.apply(state, exact_choices(p, state))
    gaps = []
    if cp["teacher"] == "COLOR_TRIALS":
        for delta in (3, 4, 5, 10, 20, 50, 100, 200):
            g = nx.disjoint_union(nx.path_graph(2), nx.star_graph(delta))
            for palette in ("global", "local"):
                p = DegreeProgram(g, cp["teacher"], 9321, palette)
                for gap in sorted({0, 1, max(1, delta // 3), delta}):
                    s = p.snapshot()
                    s.preferences[:] = 0
                    s.preferences[0, 0], s.preferences[1, 0] = delta, delta - gap
                    out = adapter(p, s)
                    exact = exact_choices(p, s)
                    # Force color zero only in a second, declared head query.
                    query = np.zeros(p.n, np.int64)
                    forced = adapter(p, s, overrides=dict(participation=s.permission, candidates=query, query=query))
                    gaps.append(dict(delta=delta, palette=palette, endpoint_palette=int(p.palette[0]),
                        gap=gap, normalized_gap=gap/delta, native_action=int(exact.color[0]),
                        predicted_action=int(out.choices.color[0]), proposal=int(out.candidates[np.where(out.roots==0)[0][0]]),
                        forced_zero_commit=bool(forced.commit[np.where(forced.roots==0)[0][0]]),
                        forced_zero_logit=float(forced.commit_logits[np.where(forced.roots==0)[0][0]])))
    result = dict(checkpoint=cp["id"], teacher=cp["teacher"], architecture=cp["architecture"], seed=cp["seed"],
        cohort=cp["cohort"], method=cp["method"], update=cp["update"], sha256=cp["sha256"],
        commit_queries={k:dict(correct=v[0], total=v[1], false_accept=v[2], false_reject=v[3], positive=v[4], negative=v[5]) for k,v in counts.items()},
        witnesses=witnesses, preference_gap_probes=gaps)
    atomic_json(ROOT/"counterfactual-probes"/f"{cp['id']}.json",result)
    return result


if __name__ == "__main__":
    cps = json.loads((ROOT/"checkpoint-inventory.json").read_text())
    cps += json.loads((ROOT/"supplemental-checkpoint-inventory.json").read_text())
    cps = [cp for cp in cps if cp["teacher"] in ("COLOR_TRIALS", "SPECULATIVE_GM")]
    atomic_json(ROOT/"counterfactual-probes/design.json",dict(namespace="degree-cf-probe-20261025",
        checkpoints=[dict(id=r['id'],sha256=r['sha256']) for r in cps],
        sources={str(p):file_hash(p) for p in [__import__('pathlib').Path(__file__),__import__('pathlib').Path('src/dcl/degree/adapter.py')]},
        interpretation="Exploratory component queries on exact safe prefixes; no autonomous success certification; different totals of labels remain a confound."))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(task,cps))
    atomic_json(ROOT/"counterfactual-probes/index.json",results)
    print(json.dumps(dict(checkpoints=len(results),queries=sum(q['total'] for r in results for q in r['commit_queries'].values()))))
