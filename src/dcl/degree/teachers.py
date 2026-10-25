"""Explicit degree/palette parameterization of six audited native rules.

No BE extension is defined here. The native schedule, RNG calls and ID space
are preserved. Local palettes are initial segments of length degree(v)+1.
"""
from __future__ import annotations

import math
import networkx as nx
import numpy as np

from dcl.multiteacher.programs import (
    ASSIGN, COLOR, DETECT, REPAIR, ID_SPACE, Choices, Snapshot, named_seed,
)

EXTENSIBLE = ("LUBY", "JOHANSSON", "COLOR_TRIALS", "JP", "SPECULATIVE_GM", "GREEDY")


def horizon(teacher, n, delta):
    if teacher == "GREEDY":
        return n
    if teacher == "SPECULATIVE_GM":
        return 2 * math.ceil(n / min(4, n)) + n
    if teacher == "JP":
        return n
    # A finite protocol cutoff, not a guarantee for a learned policy. The CT
    # bound is conservative; native runs usually finish much earlier.
    rho = (delta / (delta + 1)) ** delta / (delta + 1) if teacher == "COLOR_TRIALS" else .25
    return 128 if rho == 1 else max(128, math.ceil(math.log(n / 1e-9) / -math.log1p(-rho)))


class DegreeProgram:
    def __init__(self, graph, teacher, tape_seed, palette="global", max_rounds=None):
        if teacher not in EXTENSIBLE:
            raise ValueError("This teacher has no audited extension")
        if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
            raise ValueError("Simple undirected loop-free graphs required")
        n = len(graph)
        if not 0 < n <= ID_SPACE or set(graph) != set(range(n)):
            raise ValueError("Contiguous labels and bounded unique-ID space required")
        self.graph, self.n, self.teacher = graph, n, teacher
        self.degree = np.array([graph.degree(v) for v in range(n)], dtype=np.int64)
        self.delta = int(self.degree.max())
        if palette == "global":
            self.palette = np.full(n, self.delta + 1, dtype=np.int64)
        elif palette == "local":
            self.palette = self.degree + 1
        elif palette == "native4" and self.delta <= 3:
            self.palette = np.full(n, 4, dtype=np.int64)
        else:
            raise ValueError("Unsupported palette regime")
        self.palette_regime, self.k = palette, int(self.palette.max())
        width = max(3, self.delta)
        self.neighbors = np.full((n + 1, width), n, dtype=np.int64)
        for v in graph:
            ns = sorted(graph[v])
            self.neighbors[v, :len(ns)] = ns
        edge_list = [(u, v) for u, v in graph.edges() for u, v in ((u, v), (v, u))]
        self.edges = np.array(edge_list, dtype=np.int64).reshape(-1, 2).T
        self.rng = np.random.default_rng(tape_seed)
        self.preference_rng = np.random.default_rng(named_seed(tape_seed, "preferences"))
        self.ids = np.random.default_rng(named_seed(tape_seed, "ids")).choice(ID_SPACE, n, replace=False)
        self.order = np.argsort(self.ids)
        self.blocks = np.array_split(self.order, min(4, n))
        self.slots = np.zeros(n, np.int64)
        if teacher == "SPECULATIVE_GM":
            for block in self.blocks:
                self.slots[block] = np.arange(len(block))
        self.width = max(map(len, self.blocks))
        self.colors, self.aux, self.marks = np.full(n, -1, np.int64), np.zeros(n, np.int64), np.zeros(n, bool)
        self.phase = ASSIGN if teacher == "SPECULATIVE_GM" else COLOR
        self.index = self.steps = 0
        self.max_rounds = max_rounds or horizon(teacher, n, self.delta)
        self.repair_queue = np.array([], dtype=np.int64)
        self.finished, self.failure = False, None
        self.zero_draws = np.zeros((n, 2), np.float32)
        self.zero_preferences = np.zeros((n, self.k), np.float32)

    def snapshot(self):
        permission = self.colors < 0
        if self.teacher == "GREEDY":
            permission = np.zeros(self.n, bool)
            permission[self.order[self.index]] = True
        elif self.teacher == "SPECULATIVE_GM":
            permission = np.zeros(self.n, bool)
            if self.phase in (ASSIGN, DETECT):
                permission[[b[self.index] for b in self.blocks if self.index < len(b)]] = True
            else:
                permission[self.repair_queue[self.index]] = True
        draws = (np.stack((self.rng.random(self.n, dtype=np.float32), self.rng.random(self.n, dtype=np.float32)), -1)
                 if self.teacher in ("LUBY", "JOHANSSON") else self.zero_draws)
        # Historical CT native4 uses the historical bound even on paths/isolates.
        bound = 3 if self.palette_regime == "native4" else self.delta
        preferences = (self.preference_rng.integers(0, bound + 1, (self.n, self.k)).astype(np.float32)
                       if self.teacher == "COLOR_TRIALS" else self.zero_preferences)
        return Snapshot(self.colors.copy(), self.aux.copy(), self.marks.copy(), self.ids,
                        self.slots, permission, draws, preferences, self.phase)

    def bad_edges(self):
        a, b = self.edges
        return int(np.sum((a < b) & (self.colors[a] >= 0) & (self.colors[a] == self.colors[b])))

    def apply(self, state, choices):
        writable = state.permission
        if state.phase == DETECT:
            self.marks[writable] = choices.mark[writable]
            self.index += 1
            if self.index == self.width:
                self.repair_queue = self.order[self.marks[self.order]]
                self.phase, self.index = REPAIR, 0
                self.finished = len(self.repair_queue) == 0
        else:
            write = writable & (choices.color > 0)
            self.colors[write] = choices.color[write] - 1
            if self.teacher == "GREEDY":
                self.index += 1
                self.finished = self.index == self.n
            elif self.teacher == "SPECULATIVE_GM":
                self.index += 1
                if self.phase == ASSIGN and self.index == self.width:
                    self.phase, self.index = DETECT, 0
                elif self.phase == REPAIR:
                    self.finished = self.index == len(self.repair_queue)
            else:
                self.finished = bool(np.all(self.colors >= 0))
                if self.bad_edges():
                    self.failure, self.finished = "collision", True
        self.steps += 1
        if self.steps >= self.max_rounds and not self.finished:
            self.finished, self.failure = True, "cutoff"
        if self.finished and self.failure is None:
            if np.any((self.colors < -1) | (self.colors >= self.palette)):
                self.failure = "palette"
            elif self.bad_edges():
                self.failure = "collision"
            elif np.any(self.colors < 0):
                self.failure = "incomplete"


def availability(program, state):
    free = np.arange(program.k)[None, :] < program.palette[:, None]
    src, dst = program.edges
    occupied = state.colors[src] >= 0
    free[dst[occupied], state.colors[src[occupied]]] = False
    return free


def exact_choices(program, state):
    """Vectorized reference, never called by a neural inference adapter."""
    n, k = program.n, program.k
    mask = state.permission
    color, aux, mark = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, bool)
    src, dst = program.edges
    if state.phase == DETECT:
        conflict = ((state.colors[src] == state.colors[dst]) & (state.slots[src] == state.slots[dst]) &
                    (state.ids[dst] < state.ids[src]))
        np.logical_or.at(mark, dst, conflict)
        return Choices(color, aux, mark & mask)
    free = availability(program, state)
    if np.any(mask & ~free.any(1)):
        raise RuntimeError("No free color: reference precondition violated")
    if program.teacher in ("LUBY", "JOHANSSON"):
        rank = np.floor(state.draws[:, 1] * free.sum(1)).astype(int)
        candidate = (free & (free.cumsum(1) - 1 == rank[:, None])).argmax(1)
        part = mask & (state.draws[:, 0] < (.5 if program.teacher == "LUBY" else 1.))
        blocked = np.zeros(n, bool)
        np.logical_or.at(blocked, dst, part[src] & (candidate[src] == candidate[dst]))
        commit = part & ~blocked
        color[commit] = candidate[commit] + 1
    elif program.teacher == "JP":
        blocked = np.zeros(n, bool)
        np.logical_or.at(blocked, dst, (state.colors[src] < 0) & (state.ids[src] > state.ids[dst]))
        commit = mask & ~blocked
        color[commit] = free.argmax(1)[commit] + 1
    elif program.teacher == "COLOR_TRIALS":
        scores = np.where(free & mask[:, None], state.preferences, -1)
        other = np.full((n, k), -1, np.float32)
        # Bounded chunks avoid degree*K*neighbors materialization.
        for start in range(0, len(src), 2048):
            np.maximum.at(other, dst[start:start+2048], scores[src[start:start+2048]])
        win = free & (state.preferences > other)
        commit = mask & win.any(1)
        color[commit] = win.argmax(1)[commit] + 1
    else:
        color[mask] = free.argmax(1)[mask] + 1
    return Choices(color, aux, mark)


def scalar_choices(program, state):
    """Independent loops, including competitor-list checks for local CT."""
    out = Choices(np.zeros(program.n, np.int64), np.zeros(program.n, np.int64), np.zeros(program.n, bool))
    free = []
    for v in range(program.n):
        blocked = {int(state.colors[u]) for u in program.graph[v] if state.colors[u] >= 0}
        free.append([c for c in range(int(program.palette[v])) if c not in blocked])
    candidate, part = {}, {}
    if program.teacher in ("LUBY", "JOHANSSON"):
        for v in range(program.n):
            if state.permission[v]:
                candidate[v] = free[v][int(float(state.draws[v, 1]) * len(free[v]))]
                part[v] = bool(state.draws[v, 0] < (.5 if program.teacher == "LUBY" else 1.))
    for v in np.flatnonzero(state.permission):
        ns = list(program.graph[v])
        if state.phase == DETECT:
            out.mark[v] = any(state.colors[u] == state.colors[v] and state.slots[u] == state.slots[v]
                              and state.ids[u] > state.ids[v] for u in ns)
        elif program.teacher in ("LUBY", "JOHANSSON"):
            if part[v] and all(not part.get(u, False) or candidate[u] != candidate[v] for u in ns):
                out.color[v] = candidate[v] + 1
        elif program.teacher == "JP":
            if all(state.colors[u] >= 0 or state.ids[u] < state.ids[v] for u in ns):
                out.color[v] = min(free[v]) + 1
        elif program.teacher == "COLOR_TRIALS":
            wins = [c for c in free[v] if all(not state.permission[u] or c not in free[u] or
                    state.preferences[v, c] > state.preferences[u, c] for u in ns)]
            if wins:
                out.color[v] = min(wins) + 1
        else:
            out.color[v] = min(free[v]) + 1
    return out
