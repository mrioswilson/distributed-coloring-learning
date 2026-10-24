"""Reachability witnesses: replay admissible prefixes, never invent snapshots."""
import copy
from dataclasses import asdict

import networkx as nx
import numpy as np

from dcl.multiteacher.programs import (ASSIGN, DETECT, KW, LINIAL, REPAIR, Choices,
    ColoringProgram, available, polynomial_values, teacher_choices, MAX_FIELD)
from dcl.samplecomplexity.data import array_hash
from dcl.samplecomplexity.evaluation import invariant_after
from dcl.samplecomplexity.protocol import horizon


def state_hash(s):
    return array_hash(*(getattr(s, k) for k in ("colors", "aux", "marks", "ids", "slots",
        "permission", "draws", "preferences")), np.array([s.phase, s.field, s.bound, s.target]))


def serial_choices(c):
    return {k: getattr(c, k).tolist() for k in ("color", "auxiliary", "mark")}


def validate_choice(p, s, c):
    """Check safe protocol-prefix choices, not exact-native action fidelity."""
    n = p.n
    if any(getattr(c, k).shape != (n,) for k in ("color", "auxiliary", "mark")):
        raise ValueError("Malformed prefix action")
    if np.any(c.color[~s.permission] != 0) or np.any(c.mark[~s.permission]):
        raise ValueError("Prefix exceeds native permission")
    exact = teacher_choices(s, p.neighbors, p.teacher)
    if s.phase == DETECT:
        if not np.array_equal(c.mark, exact.mark):
            raise ValueError("Speculative marks must be recomputed from actual colors")
        return
    ns, valid = p.neighbors[:-1], p.neighbors[:-1] < n
    if s.phase == LINIAL:
        point, value = c.auxiliary // MAX_FIELD, c.auxiliary % MAX_FIELD
        if np.any((point < 0) | (point >= s.field)):
            raise ValueError("Point outside native field")
        poly = polynomial_values(s.aux, s.field)
        if not np.array_equal(value, poly[np.arange(n), point]):
            raise ValueError("Stale or incorrect polynomial value")
        other = np.vstack((poly, np.full((1, s.field), -1)))[ns]
        if np.any(valid & (other[np.arange(n)[:, None], np.arange(3), point[:, None]] == value[:, None])):
            raise ValueError("Blocked Linial point")
        return
    if np.any((c.color < 0) | (c.color > 4)):
        raise ValueError("Color action outside native alphabet")
    if s.phase == KW:
        high = s.aux >= s.target
        if not np.array_equal(c.color > 0, high):
            raise ValueError("KW must recolor exactly its high auxiliary colors")
        next_aux = s.aux.copy()
        next_aux[high] = ((s.bound - 1 - s.aux) * 4 + c.color - 1)[high]
        if np.any((next_aux < 0) | (next_aux >= s.target)) or any(next_aux[u] == next_aux[v] for u, v in p.graph.edges()):
            raise ValueError("Illegal KW continuation")
        return
    writes = c.color > 0
    if np.any(writes & ~s.permission):
        raise ValueError("Unscheduled write")
    free = available(s, p.neighbors)
    if np.any(writes & ~free[np.arange(n), np.maximum(c.color - 1, 0)]):
        raise ValueError("Prefix writes an occupied color")
    if p.teacher in ("GREEDY", "SPECULATIVE_GM"):
        if not np.array_equal(writes, s.permission):
            raise ValueError("Every scheduled native prefix write is required")
    elif p.teacher == "JP":
        if np.any(writes & (exact.color == 0)):
            raise ValueError("Priority-ineligible prefix write")
    elif p.teacher == "COLOR_TRIALS":
        prefs = np.where(free & s.permission[:, None], s.preferences, -1)
        other = np.vstack((prefs, np.full((1, 4), -1)))[ns].max(1)
        wins = free & (s.preferences > other)
        if np.any(writes & ~wins[np.arange(n), np.maximum(c.color - 1, 0)]):
            raise ValueError("Prefix uses a nonwinning ColorTrials color")
    if p.teacher != "SPECULATIVE_GM":
        if any(writes[u] and writes[v] and c.color[u] == c.color[v] for u, v in p.graph.edges()):
            raise ValueError("Simultaneous unsafe prefix")


def alternate_choice(p, s, rng):
    """Apply only safe alternatives, with dependencies supplied by the protocol."""
    c = teacher_choices(s, p.neighbors, p.teacher)
    if s.phase == DETECT:
        return c
    n = p.n
    ns = p.neighbors[:-1]
    if s.phase == LINIAL:
        poly = polynomial_values(s.aux, s.field)
        other = np.vstack((poly, np.full((1, s.field), -1)))[ns]
        legal = ~(poly[:, None] == other).any(1)
        points = np.array([rng.choice(np.flatnonzero(row)) for row in legal])
        c.auxiliary[:] = points * MAX_FIELD + poly[np.arange(n), points]
    elif s.phase == KW:
        high = s.aux >= s.target
        options = (s.bound - 1 - s.aux)[:, None] * 4 + np.arange(4)
        other = np.r_[s.aux, -2][ns]
        legal = ~(other[..., None] == options[:, None]).any(1)
        for v in np.flatnonzero(high):
            c.color[v] = int(rng.choice(np.flatnonzero(legal[v]))) + 1
    else:
        free = available(s, p.neighbors)
        if p.teacher == "COLOR_TRIALS":
            prefs = np.where(free & s.permission[:, None], s.preferences, -1)
            other = np.vstack((prefs, np.full((1, 4), -1)))[ns].max(1)
            free &= s.preferences > other
        for v in np.flatnonzero(c.color > 0):
            if rng.random() < .5:
                c.color[v] = int(rng.choice(np.flatnonzero(free[v]))) + 1
        if p.teacher in ("LUBY", "JOHANSSON"):
            # Recompute simultaneous conflicts after changing actual offers.
            bad = [(u, v) for u, v in p.graph.edges() if c.color[u] and c.color[u] == c.color[v]]
            for u, v in bad:
                c.color[u] = c.color[v] = 0
    validate_choice(p, s, c)
    return c


def trajectory(graph, teacher, tape_seed, intervention_seed=None):
    """Yield state, native target and a verifiable prefix witness.

    Replacing IDs is achieved with a fresh initialization/tape before replay.
    Fresh private draws/preferences come from the native RNG; no derived
    message or algebraic relation is edited independently.
    """
    p = ColoringProgram(graph, teacher, tape_seed, max_rounds=horizon(teacher, len(graph)))
    rng = np.random.default_rng(intervention_seed) if intervention_seed is not None else None
    prefix = []
    while not p.finished:
        s = p.snapshot()
        exact = teacher_choices(s, p.neighbors, teacher)
        witness = dict(teacher=teacher, n=len(graph), edges=[list(e) for e in sorted(graph.edges())],
            tape_seed=tape_seed, native_step=p.steps, state_sha256=state_hash(s),
            prefix=copy.deepcopy(prefix), prefix_kind="legal-alternative" if rng is not None else "exact")
        yield p, s, exact, witness
        chosen = alternate_choice(p, s, rng) if rng is not None else exact
        validate_choice(p, s, chosen)
        before, old_aux = p.colors.copy(), p.aux.copy()
        p.apply(s, chosen)
        if not invariant_after(p, s, before, old_aux):
            raise RuntimeError("Prefix invariant failed")
        prefix.append(dict(state_sha256=state_hash(s), choices=serial_choices(chosen)))
    if p.failure is not None:
        raise RuntimeError(f"Native/alternative prefix failed: {teacher}/{p.failure}")


def replay(witness):
    """Reject stale labels, edited fields and arbitrary speculative histories."""
    g = nx.Graph(); g.add_nodes_from(range(witness["n"])); g.add_edges_from(witness["edges"])
    p = ColoringProgram(g, witness["teacher"], witness["tape_seed"],
                        max_rounds=horizon(witness["teacher"], len(g)))
    for event in witness["prefix"]:
        s = p.snapshot()
        if state_hash(s) != event["state_sha256"]:
            raise ValueError("Prefix snapshot hash differs")
        raw = event["choices"]
        c = Choices(np.asarray(raw["color"], int), np.asarray(raw["auxiliary"], int), np.asarray(raw["mark"], bool))
        validate_choice(p, s, c)
        p.apply(s, c)
    if p.steps != witness["native_step"]:
        raise ValueError("Native step differs")
    s = p.snapshot()
    if state_hash(s) != witness["state_sha256"]:
        raise ValueError("Queried state differs")
    return p, s
