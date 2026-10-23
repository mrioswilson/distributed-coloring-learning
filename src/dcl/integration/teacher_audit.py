"""Independent scalar specifications and finite-domain native-teacher audit."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np

from dcl.m4_data import _edge_index
from dcl.multiteacher.programs import (ColoringProgram, Choices, TEACHERS,
    LINIAL, KW, DETECT, ASSIGN, REPAIR, teacher_choices)
from dcl.universal.rules import teacher_offer, exact_commit_labels


def polynomial(value, x, field):
    answer, power = 0, 1
    while value:
        answer = (answer + (int(value) % field) * power) % field
        value //= field
        power = power * x % field
    return answer


def scalar_choices(state, graph, teacher):
    """Loops over actual neighbors; no vectorized teacher/helper is used."""
    n = len(graph)
    color, auxiliary, mark = np.zeros(n, int), np.zeros(n, int), np.zeros(n, bool)
    free = [{c for c in range(4) if all(state.colors[u] != c for u in graph[v])} for v in graph]
    if state.phase == LINIAL:
        for v in graph:
            x = next(x for x in range(state.field)
                     if all(polynomial(state.aux[v], x, state.field) !=
                            polynomial(state.aux[u], x, state.field) for u in graph[v]))
            auxiliary[v] = 13 * x + polynomial(state.aux[v], x, state.field)
    elif state.phase == KW:
        for v in graph:
            if state.aux[v] >= state.target:
                offset = (state.bound - 1 - state.aux[v]) * 4
                color[v] = next(i + 1 for i in range(4)
                                if all(state.aux[u] != offset + i for u in graph[v]))
    elif state.phase == DETECT:
        for v in np.flatnonzero(state.permission):
            mark[v] = any(state.colors[u] == state.colors[v] and state.slots[u] == state.slots[v]
                          and state.ids[u] > state.ids[v] for u in graph[v])
    elif teacher == "GREEDY" or state.phase in (ASSIGN, REPAIR):
        for v in np.flatnonzero(state.permission):
            color[v] = min(free[v]) + 1
    elif teacher in ("LUBY", "JOHANSSON"):
        candidates, participating = {}, set()
        for v in np.flatnonzero(state.permission):
            candidates[v] = sorted(free[v])[int(float(state.draws[v, 1]) * len(free[v]))]
            if state.draws[v, 0] < (.5 if teacher == "LUBY" else 1.):
                participating.add(v)
        for v in participating:
            if all(u not in participating or candidates[u] != candidates[v] for u in graph[v]):
                color[v] = candidates[v] + 1
    elif teacher == "JP":
        for v in np.flatnonzero(state.permission):
            if all(state.colors[u] >= 0 or state.ids[v] > state.ids[u] for u in graph[v]):
                color[v] = min(free[v]) + 1
    elif teacher == "COLOR_TRIALS":
        for v in np.flatnonzero(state.permission):
            winners = [c for c in sorted(free[v]) if all(
                state.colors[u] >= 0 or c not in free[u] or state.preferences[v, c] > state.preferences[u, c]
                for u in graph[v])]
            if winners:
                color[v] = winners[0] + 1
    return Choices(color, auxiliary, mark)


def tape_differential(graph, seed):
    """Same mathematical state and explicit random arrays, not just seed IDs."""
    n, edges = len(graph), _edge_index(graph)
    tape = np.random.default_rng(seed).random((128, n, 2), dtype=np.float32)
    p = ColoringProgram(graph, "LUBY", seed)
    reference = np.full(n, -1, int)
    trace = []
    for r, draws in enumerate(tape):
        assert np.array_equal(p.colors, reference)
        state = p.snapshot(); state.draws = draws.copy()
        native = teacher_choices(state, p.neighbors, "LUBY")
        candidates, available_counts = teacher_offer(reference, draws[:, 1], edges, 4)
        participation = (reference < 0) & (draws[:, 0] < .5)
        commit = exact_commit_labels(reference, edges, participation, candidates)
        assert np.array_equal(native.color, np.where(commit, candidates + 1, 0))
        after = np.where(commit, candidates, reference)
        trace.append(dict(round=r + 1, before=reference.tolist(), draws=draws.tolist(),
                          available_counts=available_counts.tolist(), participation=participation.tolist(),
                          candidates=candidates.tolist(), commit=commit.tolist(), after=after.tolist()))
        p.apply(state, native); reference = after
        assert np.array_equal(p.colors, reference)
        if p.finished:
            break
    assert p.failure is None
    return trace


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    graphs = [nx.convert_node_labels_to_integers(g) for g in nx.graph_atlas_g()
              if 1 <= len(g) <= 6 and max(dict(g.degree()).values(), default=0) <= 3]
    rows, representative = [], {}
    for name in TEACHERS:
        executions = states = cutoffs = 0
        for graph in graphs:
            for seed in (2026102301, 2026102302):
                p = ColoringProgram(graph, name, seed)
                while not p.finished:
                    state = p.snapshot()
                    got, expected = teacher_choices(state, p.neighbors, name), scalar_choices(state, graph, name)
                    for a, b in zip((got.color, got.auxiliary, got.mark),
                                    (expected.color, expected.auxiliary, expected.mark)):
                        assert np.array_equal(a, b), (name, list(graph.edges()), state.phase)
                    before = p.colors.copy(); p.apply(state, got); states += 1
                    assert np.all((p.colors >= -1) & (p.colors < 4))
                    if name not in ("SPECULATIVE_GM", "BE_BASE_D3"):
                        assert np.array_equal(before[before >= 0], p.colors[before >= 0])
                        assert p.bad_edges() == 0
                    if name == "BE_BASE_D3":
                        assert all(p.aux[u] != p.aux[v] for u, v in graph.edges())
                    if len(graph) == 4 and nx.is_isomorphic(graph, nx.path_graph(4)) and seed == 2026102301:
                        representative.setdefault(name, []).append(dict(
                            phase=state.phase, before=before.tolist(), auxiliary_before=state.aux.tolist(),
                            participation=state.permission.tolist(), draws=state.draws.tolist(),
                            preferences=state.preferences.tolist(), ids=state.ids.tolist(),
                            choices=got.color.tolist(), auxiliary_choices=got.auxiliary.tolist(),
                            marks=got.mark.tolist(), after=p.colors.tolist(), auxiliary_after=p.aux.tolist()))
                executions += 1
                cutoffs += p.failure == "incomplete"
                assert p.failure in (None, "incomplete")
                if p.failure is None:
                    assert (p.colors >= 0).all() and p.bad_edges() == 0
        rows.append(dict(teacher=name, graphs=len(graphs), executions=executions,
                         scalar_transition_comparisons=states, cutoffs=cutoffs))
    differential = []
    for name, graph in [("path8", nx.path_graph(8)), ("star3", nx.star_graph(3)),
                         ("clique4", nx.complete_graph(4)),
                         ("cubic40", nx.random_regular_graph(3, 40, seed=20261023))]:
        for seed in (2026102303, 2026102304):
            trace = tape_differential(graph, seed)
            differential.append(dict(graph=name, seed=seed, rounds=len(trace), passed=True))
            if name == "star3":
                (output / f"shared-tape-{name}-{seed}.json").write_text(json.dumps(trace, indent=2) + "\n")
    (output / "native-representative-traces.json").write_text(json.dumps(representative, indent=2) + "\n")
    result = dict(passed=True, finite_graph_domain="Every unlabeled simple graph with 1..6 vertices and maximum degree <=3, two specified tapes each; not all IDs or tapes.",
                  teachers=rows, same_tape_differentials=differential,
                  correctness_scope="Scalar specifications agree on this finite domain. Universal guarantees require the separate conditional proofs; finite traces do not prove universal correctness.")
    (output / "teacher-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="results/integration/teachers")
    run(parser.parse_args().output)
