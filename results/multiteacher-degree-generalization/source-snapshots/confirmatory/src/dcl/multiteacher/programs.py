"""Coloring programs for maximum degree three and a common four-color palette.

Sources and exact scope:
* JP: fixed distinct priorities, local maxima, first free color.
* COLOR_TRIALS: Schneider--Wattenhofer ColorTrials, repeated; global degree
  bound for the discrete preferences. This does not implement RankingTrials.
* BE_BASE_D3: Delta-Color's i=log*3=1 base (KW Theorem 5.1), initialized by
  polynomial Linial reduction. It is not an implementation of the high-degree
  recursion or of the sharper Szegedy--Vishwanathan preprocessing constant.
* SPECULATIVE_GM: Gebremedhin--Manne Algorithm 1, four balanced blocks,
  synchronized first fit, same-slot conflict detection, serial repair.
* GREEDY: sequential first fit in a random permutation.

The interpreter supplies native phases and sequential schedules. Predictions
replace the local choices, including conflict marking and auxiliary colors.
Every predicted color choice is applied; no validity rule filters predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import networkx as nx
import numpy as np

TEACHERS = ("LUBY", "JOHANSSON", "JP", "COLOR_TRIALS", "BE_BASE_D3", "SPECULATIVE_GM", "GREEDY")
COLOR, ASSIGN, DETECT, REPAIR, LINIAL, KW = range(6)
K, DEGREE, ID_SPACE, MAX_FIELD = 4, 3, 65536, 13


def named_seed(*parts):
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:16], 16)


def polynomial_values(values, field):
    """Base-field coefficients, evaluated at every element of a prime field."""
    values = np.asarray(values, dtype=np.int64).copy()
    result = np.zeros(values.shape + (field,), dtype=np.int64)
    power = np.ones_like(result)
    points = np.arange(field)
    while np.any(values):
        result = (result + (values % field)[..., None] * power) % field
        values //= field
        power = power * points % field
    return result


def linial_plan():
    # p > degree * polynomial_degree ensures at least one unblocked point.
    bound, plan = ID_SPACE, []
    while bound > 16 * DEGREE**2:
        for field in (2, 3, 5, 7, 11, 13):
            power, degree = field, 0
            while power < bound:
                power *= field
                degree += 1
            if field > DEGREE * degree:
                break
        else:
            raise RuntimeError("No supported field for the declared identity space")
        plan.append((bound, field))
        bound = field**2
    return tuple(plan), bound


@dataclass
class Snapshot:
    colors: np.ndarray
    aux: np.ndarray
    marks: np.ndarray
    ids: np.ndarray
    slots: np.ndarray
    permission: np.ndarray
    draws: np.ndarray
    preferences: np.ndarray
    phase: int
    field: int = 13
    bound: int = ID_SPACE
    target: int = 0


@dataclass
class Choices:
    color: np.ndarray  # 0=WAIT; 1..4=write colors 0..3
    auxiliary: np.ndarray  # pair x,y encoded as x*MAX_FIELD+y
    mark: np.ndarray


class ColoringProgram:
    def __init__(self, graph, teacher, tape_seed, max_rounds=128):
        if teacher not in TEACHERS:
            raise ValueError("Unknown teacher")
        if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
            raise ValueError("A simple undirected loop-free graph is required")
        n = len(graph)
        if set(graph) != set(range(n)) or not n or n > ID_SPACE:
            raise ValueError("Contiguous labels and 1..65536 vertices are required")
        if max(dict(graph.degree()).values(), default=0) > DEGREE:
            raise ValueError("This explicit BE base-case benchmark supports degree <=3")
        self.graph, self.teacher, self.n = graph, teacher, n
        self.neighbors = np.full((n + 1, DEGREE), n, dtype=np.int64)
        for v in graph:
            ns = sorted(graph.neighbors(v))
            self.neighbors[v, :len(ns)] = ns
        self.rng = np.random.default_rng(tape_seed)
        self.preference_rng = np.random.default_rng(named_seed(tape_seed, "preferences"))
        self.ids = np.random.default_rng(named_seed(tape_seed, "ids")).choice(ID_SPACE, n, replace=False)
        self.order = np.argsort(self.ids)
        self.blocks = np.array_split(self.order, min(4, n))
        self.slots = np.zeros(n, dtype=np.int64)
        # A global partition position belongs only to the native speculative
        # schedule. Supplying it to parallel teachers would leak a nonlocal
        # preprocessing result into an otherwise local decision rule.
        if teacher == "SPECULATIVE_GM":
            for block in self.blocks:
                self.slots[block] = np.arange(len(block))
        self.width = max(map(len, self.blocks))
        self.colors = np.full(n, -1, dtype=np.int64)
        self.aux = self.ids.copy() if teacher == "BE_BASE_D3" else np.zeros(n, dtype=np.int64)
        self.marks = np.zeros(n, dtype=bool)
        self.phase = LINIAL if teacher == "BE_BASE_D3" else (ASSIGN if teacher == "SPECULATIVE_GM" else COLOR)
        self.index, self.steps, self.max_rounds = 0, 0, max_rounds
        self.preparation, self.bound = linial_plan()
        self.repair_queue = np.array([], dtype=np.int64)
        self.finished = False
        self.failure = None
        self.zero_draws = np.zeros((n, 2), np.float32)
        self.zero_preferences = np.zeros((n, K), np.float32)

    def snapshot(self, copy_arrays=True):
        if self.finished:
            raise RuntimeError("Program is already finished")
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
        elif self.teacher == "BE_BASE_D3":
            permission = np.ones(self.n, bool)
        field = self.preparation[self.index][1] if self.phase == LINIAL else MAX_FIELD
        bound = self.preparation[self.index][0] if self.phase == LINIAL else self.bound
        target = math.ceil(bound * K / (K + 1)) if self.phase == KW else 0
        # The native random inputs are given to the network; the teacher also
        # uses exactly these float32 values, avoiding encoding/label mismatch.
        draws = (np.stack((self.rng.random(self.n, dtype=np.float32), self.rng.random(self.n, dtype=np.float32)), axis=-1)
                 if self.teacher in ("LUBY", "JOHANSSON") else self.zero_draws)
        preferences = (self.preference_rng.integers(0, DEGREE + 1, size=(self.n, K)).astype(np.float32)
                       if self.teacher == "COLOR_TRIALS" else self.zero_preferences)
        colors, aux, marks = ((a.copy() if copy_arrays else a) for a in (self.colors, self.aux, self.marks))
        return Snapshot(colors, aux, marks, self.ids,
                        self.slots, permission, draws, preferences, self.phase, field, bound, target)

    def apply(self, state, choices):
        """Only interpreter permissions are enforced, never coloring validity."""
        writable = state.permission
        if state.phase == LINIAL:
            x, y = choices.auxiliary // MAX_FIELD, choices.auxiliary % MAX_FIELD
            self.aux[writable] = (x * state.field + y)[writable]
            self.index += 1
            if self.index == len(self.preparation):
                self.phase, self.index = KW, 0
        elif state.phase == KW:
            write = writable & (choices.color > 0)
            offsets = (state.bound - 1 - state.aux) * K
            self.aux[write] = (offsets + choices.color - 1)[write]
            self.bound = state.target
            if self.bound == K:
                self.colors = self.aux.copy()
                self.finished = True
        elif state.phase == DETECT:
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
        if self.teacher in ("LUBY", "JOHANSSON", "JP", "COLOR_TRIALS") and self.steps >= self.max_rounds:
            self.finished = True
        if self.finished and self.failure is None:
            if np.any((self.colors < -1) | (self.colors >= K)):
                self.failure = "palette"
            elif self.bad_edges():
                self.failure = "collision"
            elif np.any(self.colors < 0):
                self.failure = "incomplete"

    def bad_edges(self):
        return sum(self.colors[u] >= 0 and self.colors[u] == self.colors[v] for u, v in self.graph.edges())


def available(state, neighbors):
    pad = np.r_[state.colors, -2]
    neighbor_colors = pad[neighbors[:-1]]
    return ~np.any(neighbor_colors[..., None] == np.arange(K), axis=1)


def teacher_choices(state, neighbors, teacher):
    n = len(state.colors)
    mask = state.permission
    color = np.zeros(n, np.int64)
    auxiliary = np.zeros(n, np.int64)
    mark = np.zeros(n, bool)
    ns = neighbors[:n]
    valid_neighbor = ns < n
    if teacher == "GREEDY" or state.phase in (ASSIGN, REPAIR, DETECT):
        roots = np.flatnonzero(mask)
        adjacent = neighbors[roots]
        valid = adjacent < n
        clipped = np.minimum(adjacent, n - 1)
        if state.phase == DETECT:
            conflict = valid & (state.colors[clipped] == state.colors[roots, None])
            conflict &= state.slots[clipped] == state.slots[roots, None]
            conflict &= state.ids[roots, None] < state.ids[clipped]
            mark[roots] = conflict.any(1)
        else:
            blocked = valid[..., None] & (state.colors[clipped, None] == np.arange(K))
            free = ~blocked.any(1)
            color[roots] = free.argmax(1) + 1
        return Choices(color, auxiliary, mark)
    if state.phase == LINIAL:
        poly = polynomial_values(state.aux, state.field)
        other = np.vstack((poly, np.full((1, state.field), -1)))[ns]
        forbidden = np.any((poly[:, None, :] == other) & valid_neighbor[..., None], axis=1)
        if np.any(forbidden.all(1)):
            raise RuntimeError("Linial precondition failed: input auxiliary coloring is not proper")
        x = (~forbidden).argmax(1)
        auxiliary = x * MAX_FIELD + poly[np.arange(n), x]
    elif state.phase == KW:
        recolor = state.aux >= state.target
        offsets = (state.bound - 1 - state.aux) * K
        options = offsets[:, None] + np.arange(K)
        other = np.r_[state.aux, -2][ns]
        free = ~np.any(other[..., None] == options[:, None, :], axis=1)
        if np.any(recolor & ~free.any(1)):
            raise RuntimeError("KW precondition failed")
        color[recolor] = free.argmax(1)[recolor] + 1
    elif state.phase == DETECT:
        neighbor_colors = np.r_[state.colors, -2][ns]
        neighbor_ids = np.r_[state.ids, -1][ns]
        neighbor_slots = np.r_[state.slots, -1][ns]
        mark = mask & np.any(valid_neighbor & (neighbor_colors == state.colors[:, None]) &
                             (neighbor_slots == state.slots[:, None]) & (state.ids[:, None] < neighbor_ids), axis=1)
    else:
        free = available(state, neighbors)
        if np.any(mask & ~free.any(1)):
            raise RuntimeError("Four colors must suffice at maximum degree three")
        if teacher in ("LUBY", "JOHANSSON"):
            rank = np.floor(state.draws[:, 1] * free.sum(1)).astype(int)
            candidates = (free & (free.cumsum(1) - 1 == rank[:, None])).argmax(1)
            part = mask & (state.draws[:, 0] < (0.5 if teacher == "LUBY" else 1.0))
            neighbor_part = np.r_[part, False][ns]
            neighbor_candidate = np.r_[candidates, -2][ns]
            blocked = np.any(neighbor_part & (neighbor_candidate == candidates[:, None]), axis=1)
            commit = part & ~blocked
            color[commit] = candidates[commit] + 1
        elif teacher == "JP":
            neighbor_pending = np.r_[state.colors < 0, False][ns]
            neighbor_ids = np.r_[state.ids, -1][ns]
            commit = mask & ~np.any(neighbor_pending & (neighbor_ids > state.ids[:, None]), axis=1)
            color[commit] = free.argmax(1)[commit] + 1
        elif teacher == "COLOR_TRIALS":
            # A neighbor competes for c only when c belongs to its palette.
            scores = np.where(free & mask[:, None], state.preferences, -1)
            other = np.concatenate((scores, np.full((1, K), -1)), axis=0)[ns]
            winning = free & (state.preferences > other.max(1))
            commit = mask & winning.any(1)
            color[commit] = winning.argmax(1)[commit] + 1
        else:
            color[mask] = free.argmax(1)[mask] + 1
    return Choices(color, auxiliary, mark)


def run_teacher(graph, teacher, tape_seed, save_trace=False):
    program = ColoringProgram(graph, teacher, tape_seed)
    trace = []
    while not program.finished:
        state = program.snapshot()
        choices = teacher_choices(state, program.neighbors, teacher)
        if save_trace:
            trace.append((state, choices))
        program.apply(state, choices)
    return {"success": program.failure is None, "failure": program.failure,
            "steps": program.steps, "colors": program.colors.copy(), "trace": trace}
