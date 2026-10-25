"""Frozen autonomous execution and read-only divergence diagnostics."""
from collections import defaultdict
from types import SimpleNamespace
import time
import numpy as np
import torch
from dcl.multiteacher.programs import Choices, Snapshot, DETECT
from .teachers import DegreeProgram, exact_choices, availability
from .adapter import Decision


def combine(items):
    programs, states = zip(*items)
    sizes = [p.n for p in programs]
    offsets = np.r_[0, np.cumsum(sizes)]
    total = int(offsets[-1])
    width = max(p.neighbors.shape[1] for p in programs)
    neighbors = np.full((total + 1, width), total, np.int64)
    edges = []
    for i, p in enumerate(programs):
        a = p.neighbors[:p.n]
        neighbors[offsets[i]:offsets[i+1], :a.shape[1]] = np.where(a < p.n, a + offsets[i], total)
        edges.append(p.edges + offsets[i])
    program = SimpleNamespace(n=total, k=programs[0].k, teacher=programs[0].teacher,
        delta=programs[0].delta, palette_regime=programs[0].palette_regime,
        palette=np.concatenate([p.palette for p in programs]), neighbors=neighbors,
        edges=np.concatenate(edges, 1))
    fields = ("colors", "aux", "marks", "ids", "slots", "permission", "draws", "preferences")
    state = Snapshot(*(np.concatenate([getattr(s, f) for s in states]) for f in fields), phase=states[0].phase)
    return program, state, offsets


def split_decision(decision, offsets):
    result = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        use = (decision.roots >= lo) & (decision.roots < hi)
        result.append(Decision(Choices(decision.choices.color[lo:hi], decision.choices.auxiliary[lo:hi],
                                       decision.choices.mark[lo:hi]), decision.roots[use] - lo,
            *(getattr(decision, f)[use] for f in ("participation", "candidates", "commit", "p_logits",
                                                 "commit_logits", "availability", "scores"))))
    return result


def component_diagnostics(p, s, d):
    roots, teacher = d.roots, p.teacher
    if s.phase == DETECT:
        truth = exact_choices(p, s).mark[roots]
        return {"conflict_mark": (d.commit, truth)}
    free = availability(p, s)
    c = free.argmax(1)
    active = s.permission.copy()
    src, dst = p.edges
    if teacher in ("LUBY", "JOHANSSON"):
        rank = np.floor(s.draws[:, 1] * free.sum(1)).astype(int)
        c = (free & (free.cumsum(1) - 1 == rank[:, None])).argmax(1)
        active &= s.draws[:, 0] < (.5 if teacher == "LUBY" else 1.)
    elif teacher == "JP":
        bad = np.zeros(p.n, bool)
        np.logical_or.at(bad, dst, (s.colors[src] < 0) & (s.ids[src] > s.ids[dst]))
        active &= ~bad
    winning = None
    if teacher == "COLOR_TRIALS":
        prefs = np.where(free & s.permission[:, None], s.preferences, -1)
        other = np.full((p.n, p.k), -1, np.float32)
        np.maximum.at(other, dst, prefs[src])
        winning = free & (s.preferences > other)
        c = np.where(winning.any(1), winning.argmax(1), c)
    # Confirmation is re-queried on the actual neural proposals. This oracle
    # is a measurement, and is never used as the applied action.
    ps, cs = np.zeros(p.n, bool), np.zeros(p.n, np.int64)
    ps[roots], cs[roots] = d.participation, d.candidates
    confirm = ps & s.permission & free[np.arange(p.n), cs]
    if teacher == "JP":
        confirm &= active
    if teacher in ("LUBY", "JOHANSSON", "JP"):
        blocked = np.zeros(p.n, bool)
        np.logical_or.at(blocked, dst, ps[src] & (cs[src] == cs[dst]))
        confirm &= ~blocked
    if winning is not None:
        confirm &= winning[np.arange(p.n), cs]
    return {"availability": (d.availability > .5, free[roots]),
            "participation": (d.participation, active[roots]),
            "proposal": (d.candidates, c[roots]), "commit_matched": (d.commit, confirm[roots])}


def trace_state(p, s, d, exact, component):
    return dict(step=p.steps, phase=s.phase, first_component=component,
        colors=s.colors.tolist(), ids=s.ids.tolist(), slots=s.slots.tolist(),
        permission=s.permission.tolist(), draws=s.draws.tolist(), preferences=s.preferences.tolist(),
        palette=p.palette.tolist(), roots=d.roots.tolist(), predicted_color=d.choices.color.tolist(),
        predicted_mark=d.choices.mark.tolist(), exact_color=exact.color.tolist(), exact_mark=exact.mark.tolist(),
        participation=d.participation.tolist(), candidates=d.candidates.tolist(), commit=d.commit.tolist(),
        p_logits=d.p_logits.tolist(), commit_logits=d.commit_logits.tolist(),
        availability=d.availability.tolist(), scores=d.scores.tolist())


@torch.inference_mode()
def evaluate(adapter, teacher, cases, palette, batch=16, diagnostics=True, trace_limit=1, wall_budget_seconds=None):
    rows, traces = [], []
    wall, cpu = time.perf_counter(), time.process_time()
    for start in range(0, len(cases), batch):
        batch_started = time.perf_counter()
        part = cases[start:start+batch]
        programs = [DegreeProgram(g, teacher, m["tape_seed"], palette) for g, m in part]
        stats = [dict(action_correct=0, action_total=0, transitions=0, same_transitions=0,
                      first_divergence=None, first_action_divergence=None, components=defaultdict(lambda: [0, 0]), invariants=True,
                      min_margin=None) for _ in part]
        first_traces = [None] * len(part)
        while any(not p.finished for p in programs):
            if wall_budget_seconds is not None and time.perf_counter() - batch_started >= wall_budget_seconds:
                for p in programs:
                    if not p.finished:
                        p.finished, p.failure = True, "resource_cutoff"
                break
            groups = defaultdict(list)
            for i, p in enumerate(programs):
                if not p.finished:
                    groups[(p.phase, p.k)].append((i, p, p.snapshot()))
            for group in groups.values():
                if adapter is None:
                    decisions = [None] * len(group)
                else:
                    joined, state, offsets = combine([(p, s) for _, p, s in group])
                    decisions = split_decision(adapter(joined, state), offsets)
                for (i, p, state), decision in zip(group, decisions):
                    st = stats[i]
                    exact = exact_choices(p, state) if diagnostics or adapter is None else None
                    choice = exact if adapter is None else decision.choices
                    if adapter is not None and diagnostics:
                        r = decision.roots
                        truth = exact.mark[r] if state.phase == DETECT else exact.color[r]
                        pred = choice.mark[r] if state.phase == DETECT else choice.color[r]
                        equal = pred == truth
                        st["action_correct"] += int(equal.sum())
                        st["action_total"] += len(r)
                        st["transitions"] += 1
                        st["same_transitions"] += int(equal.all())
                        components = component_diagnostics(p, state, decision)
                        first = None
                        for name, (prediction, target) in components.items():
                            use = np.ones_like(target, bool)
                            if name == "availability":
                                use = np.arange(p.k)[None] < p.palette[r, None]
                            st["components"][name][0] += int(((prediction == target) & use).sum())
                            st["components"][name][1] += int(use.sum())
                            if first is None and np.any((prediction != target) & use):
                                first = name
                        margin = float(min(np.abs(decision.p_logits).min(), np.abs(decision.commit_logits).min()))
                        st["min_margin"] = margin if st["min_margin"] is None else min(st["min_margin"], margin)
                        if first and st["first_divergence"] is None:
                            st["first_divergence"] = dict(step=p.steps, phase=state.phase, component=first,
                                actual_action_differs=not bool(equal.all()))
                            if len(traces) < trace_limit:
                                first_traces[i] = trace_state(p, state, decision, exact, first)
                        if not bool(equal.all()) and st["first_action_divergence"] is None:
                            st["first_action_divergence"] = dict(step=p.steps, phase=state.phase,
                                roots=r[~equal].tolist(), component=first)
                            if len(traces) < trace_limit:
                                first_traces[i] = dict(first_component_trace=first_traces[i],
                                    first_action_trace=trace_state(p, state, decision, exact, first))
                    before = p.colors.copy()
                    p.apply(state, choice)
                    palette_ok = bool(np.all((p.colors >= -1) & (p.colors < p.palette)))
                    unchanged = bool(np.array_equal(before[~state.permission], p.colors[~state.permission]))
                    invariant = palette_ok and unchanged
                    if teacher != "SPECULATIVE_GM":
                        invariant &= not p.bad_edges() and np.array_equal(before[before >= 0], p.colors[before >= 0])
                    st["invariants"] &= bool(invariant)
                    if not invariant:
                        p.failure, p.finished = p.failure or "invariant", True
        for (g, meta), p, st, tr in zip(part, programs, stats, first_traces):
            conflicts = p.bad_edges()
            palette_violations = int(np.sum((p.colors < -1) | (p.colors >= p.palette)))
            incomplete = int(np.sum(p.colors < 0))
            success = not conflicts and not palette_violations and not incomplete and st["invariants"] and p.failure is None
            row = dict(**{k: v for k, v in meta.items() if k != "edges"}, teacher=teacher, palette=palette,
                success=bool(success), failure=p.failure, conflicts=conflicts, palette_violations=palette_violations,
                incomplete=incomplete, cutoff=p.failure in ("cutoff", "resource_cutoff"), steps=p.steps, horizon=p.max_rounds,
                first_divergence=st["first_divergence"], invariants=st["invariants"], colors=p.colors.tolist(),
                first_action_divergence=st["first_action_divergence"],
                action_correct=st["action_correct"], action_total=st["action_total"],
                transitions=st["transitions"], same_transitions=st["same_transitions"],
                components=dict(st["components"]), min_margin=st["min_margin"])
            rows.append(row)
            if not success and tr is not None and len(traces) < trace_limit:
                traces.append(dict(case=meta, trace=tr, final=row))
    return rows, traces, dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                             wall_budget_seconds=wall_budget_seconds)
