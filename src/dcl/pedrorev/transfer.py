"""Frozen transfer with on-policy contract accounting (Experiment D).

Same execution semantics as `dcl.degree.evaluation.evaluate` (the applied
action is always the network's; the contract oracle is only a measurement).
In addition, at every color-phase step and permitted root it counts, against
the declared confirmation contract evaluated on the network's *own* messages:

* bad proposals: participates, but the contract would reject its proposal;
* unsafe accepts: the network writes although the contract rejects;
* contained: bad proposal and the network's COMMIT rejects it;
* missed: the contract would accept, the network participates but rejects.

A failed run with at least one unsafe accept is classified as a safety
(accepted-violation) failure; otherwise as a progress failure (cutoff, missed
scheduled write, or a speculative conflict never repaired).
"""
import time

import numpy as np
import torch

from dcl.degree.adapter import FrozenAdapter
from dcl.degree.evaluation import combine, split_decision
from dcl.degree.graphs import make_case
from dcl.degree.teachers import DegreeProgram, availability, exact_choices
from dcl.multiteacher.programs import DETECT, named_seed
from dcl.samplecomplexity.model import QueryPolicy

COUNTERS = ("bad_proposals", "unsafe_accepts", "contained", "missed", "participations", "decisions")


def contract_accepts(p, s, roots, part, cand):
    """Declared confirmation contract on the network's own messages (measurement only)."""
    free = availability(p, s)
    ps, cs = np.zeros(p.n, bool), np.zeros(p.n, np.int64)
    ps[roots], cs[roots] = part, cand
    ok = ps & s.permission & free[np.arange(p.n), np.minimum(cs, free.shape[1] - 1)] & (cs < free.shape[1])
    src, dst = p.edges
    if p.teacher == "JP":
        bad = np.zeros(p.n, bool)
        np.logical_or.at(bad, dst, (s.colors[src] < 0) & (s.ids[src] > s.ids[dst]))
        ok &= ~bad
    if p.teacher in ("LUBY", "JOHANSSON", "JP"):
        blocked = np.zeros(p.n, bool)
        np.logical_or.at(blocked, dst, ps[src] & (cs[src] == cs[dst]))
        ok &= ~blocked
    if p.teacher == "COLOR_TRIALS":
        prefs = np.where(free & s.permission[:, None], s.preferences, -1)
        other = np.full((p.n, p.k), -1, np.float32)
        np.maximum.at(other, dst, prefs[src])
        winning = free & (s.preferences > other)
        ok &= winning[np.arange(p.n), np.minimum(cs, p.k - 1)]
    return ok[roots]


def suite(namespace, condition):
    out = []
    for i in range(condition["runs"]):
        index = int(np.random.default_rng(named_seed(namespace, "family", condition["n"], condition["delta"], i))
                    .integers(len(condition["families"])))
        out.append(make_case(namespace, condition["n"], condition["delta"], condition["families"][index], i, 0))
    return out


@torch.inference_mode()
def run(adapter, teacher, cases, palette, batch=16, last_state=False):
    rows = []
    tick = time.process_time()
    for start in range(0, len(cases), batch):
        part = cases[start:start + batch]
        programs = [DegreeProgram(g, teacher, m["tape_seed"], palette) for g, m in part]
        stats = [dict.fromkeys(COUNTERS, 0) | dict(first_unsafe=None, invariants=True) for _ in part]
        while any(not p.finished for p in programs):
            groups = {}
            for i, p in enumerate(programs):
                if not p.finished:
                    groups.setdefault((p.phase, p.k), []).append((i, p, p.snapshot()))
            for group in groups.values():
                if adapter is None:
                    decisions = [None] * len(group)
                else:
                    joined, state, offsets = combine([(p, s) for _, p, s in group])
                    decisions = split_decision(adapter(joined, state), offsets)
                for (i, p, s), d in zip(group, decisions):
                    choice = exact_choices(p, s) if adapter is None else d.choices
                    if adapter is not None and s.phase != DETECT:
                        st = stats[i]
                        ok = contract_accepts(p, s, d.roots, d.participation, d.candidates)
                        write = d.participation & d.commit
                        st["decisions"] += len(d.roots)
                        st["participations"] += int(d.participation.sum())
                        st["bad_proposals"] += int((d.participation & ~ok).sum())
                        unsafe = int((write & ~ok).sum())
                        st["unsafe_accepts"] += unsafe
                        st["contained"] += int((d.participation & ~ok & ~d.commit).sum())
                        st["missed"] += int((d.participation & ok & ~d.commit).sum())
                        if unsafe and st["first_unsafe"] is None:
                            st["first_unsafe"] = dict(step=p.steps, phase=int(s.phase))
                    if last_state and adapter is not None and s.phase != DETECT:
                        pend = d.roots[(s.colors[d.roots] < 0)] if p.teacher != "SPECULATIVE_GM" else d.roots
                        free = availability(p, s)
                        stats[i]["last"] = [dict(vertex=int(v), available=np.flatnonzero(free[v]).tolist(),
                                                 participates=bool(d.participation[k]), candidate=int(d.candidates[k]),
                                                 commit=bool(d.commit[k]))
                                            for k, v in enumerate(d.roots) if v in set(pend.tolist())][:12]
                    before = p.colors.copy()
                    p.apply(s, choice)
                    palette_ok = bool(np.all((p.colors >= -1) & (p.colors < p.palette)))
                    unchanged = bool(np.array_equal(before[~s.permission], p.colors[~s.permission]))
                    invariant = palette_ok and unchanged
                    if teacher != "SPECULATIVE_GM":
                        invariant &= not p.bad_edges() and np.array_equal(before[before >= 0], p.colors[before >= 0])
                    stats[i]["invariants"] &= bool(invariant)
                    if not invariant:
                        p.failure, p.finished = p.failure or "invariant", True
        for (g, meta), p, st in zip(part, programs, stats):
            conflicts = p.bad_edges()
            incomplete = int(np.sum(p.colors < 0))
            palette_violations = int(np.sum((p.colors < -1) | (p.colors >= p.palette)))
            success = not conflicts and not incomplete and not palette_violations and st["invariants"] and p.failure is None
            kind = None if success else ("safety" if st["unsafe_accepts"] else "progress")
            rows.append(dict(case_id=meta["case_id"], family=meta["family"], n=meta["n"], delta=meta["delta"],
                             success=bool(success), failure=p.failure, failure_kind=kind, conflicts=int(conflicts),
                             incomplete=incomplete, steps=int(p.steps), **{k: st[k] for k in COUNTERS},
                             first_unsafe=st["first_unsafe"], last=st.get("last")))
    return rows, time.process_time() - tick


def load_adapter(teacher, architecture, checkpoint):
    model = QueryPolicy(teacher, architecture)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    return FrozenAdapter(model)
