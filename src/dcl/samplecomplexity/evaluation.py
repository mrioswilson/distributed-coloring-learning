"""Autonomous rollouts, explicit native invariants, no repairs and no retries."""
from collections import defaultdict
import time

import numpy as np
import torch

from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, Choices, KW, LINIAL, DETECT, teacher_choices
from dcl.multiteacher_e1.experiment import predictions
from .protocol import horizon, passed_run


def invariant_after(program, state, before, old_aux):
    if program.teacher == "BE_BASE_D3":
        bound = state.field ** 2 if state.phase == LINIAL else state.target
        return bool(np.all((program.aux >= 0) & (program.aux < bound)) and
                    all(program.aux[u] != program.aux[v] for u, v in program.graph.edges()))
    if program.teacher == "SPECULATIVE_GM":
        # Conflicts during speculation are legal. Only writable nodes may change.
        return bool(np.array_equal(before[~state.permission], program.colors[~state.permission]))
    fixed = before >= 0
    return bool(np.array_equal(before[fixed], program.colors[fixed]) and program.bad_edges() == 0)


@torch.inference_mode()
def evaluate(model, teacher, cases, batch_size=1024, cohort=32, fidelity=False):
    """Every specified case produces a row, including technical exceptions."""
    rows = []; tick = time.perf_counter()
    if model is not None: model.eval()
    for offset in range(0, len(cases), cohort):
        part = cases[offset:offset+cohort]
        programs = [ColoringProgram(g, teacher, m["tape_seed"], max_rounds=horizon(teacher, len(g))) for g, m in part]
        stats = [dict(invariants=True, technical_error=None, cutoff=False,
                      action_correct=0, action_total=0) for _ in programs]
        while any(not p.finished for p in programs):
            pending = []
            for i, p in enumerate(programs):
                if p.finished: continue
                if p.steps >= horizon(teacher, p.n):
                    stats[i]["cutoff"] = True; p.finished = True; p.failure = "cutoff"; continue
                try:
                    pending.append((i, p, p.snapshot()))
                except Exception as error:
                    stats[i]["technical_error"] = f"{type(error).__name__}: {error}"
                    p.failure = "technical_error"; p.finished = True
            all_arrays, destinations, phases = [], [], []
            choices = {}
            for i, p, state in pending:
                roots = np.flatnonzero(state.permission)
                choices[i] = Choices(np.zeros(p.n, int), np.zeros(p.n, int), np.zeros(p.n, bool))
                if model is None:
                    try:
                        choices[i] = teacher_choices(state, p.neighbors, teacher)
                    except Exception as error:
                        stats[i]["technical_error"] = f"{type(error).__name__}: {error}"
                        p.finished = True; p.failure = "technical_error"
                else:
                    try:
                        all_arrays.append(encode_roots(state, p.neighbors, roots))
                        destinations.append((i, roots)); phases.extend([state.phase]*len(roots))
                    except Exception as error:
                        stats[i]["technical_error"] = f"{type(error).__name__}: {error}"
                        p.finished = True; p.failure = "technical_error"
            if all_arrays:
                arrays = tuple(np.concatenate([a[j] for a in all_arrays]) for j in range(4))
                phase_array = np.asarray(phases)
                try:
                    chunks = [predictions(model, tuple(a[start:start+batch_size] for a in arrays), phase_array[start:start+batch_size])
                              for start in range(0, len(phases), batch_size)]
                    predicted = tuple(np.concatenate([c[j] for c in chunks]) for j in range(3))
                    cursor = 0
                    for i, roots in destinations:
                        for dest, pred in zip((choices[i].color, choices[i].auxiliary, choices[i].mark), predicted):
                            dest[roots] = pred[cursor:cursor+len(roots)]
                        cursor += len(roots)
                except Exception as error:
                    for i, _ in destinations:
                        stats[i]["technical_error"] = f"{type(error).__name__}: {error}"
                        programs[i].finished = True; programs[i].failure = "technical_error"
            for i, p, state in pending:
                if p.finished: continue
                before, old_aux = p.colors.copy(), p.aux.copy()
                try:
                    if fidelity and model is not None:
                        # Only admissible states; this is an evaluation diagnostic,
                        # never a training acquisition or an inference shield.
                        from .interactive import admissible
                        if admissible(p, state):
                            exact = teacher_choices(state, p.neighbors, teacher)
                            field = "auxiliary" if state.phase == LINIAL else ("mark" if state.phase == DETECT else "color")
                            use = state.permission
                            stats[i]["action_total"] += int(use.sum())
                            stats[i]["action_correct"] += int((getattr(exact, field)[use] == getattr(choices[i], field)[use]).sum())
                    p.apply(state, choices[i])
                    valid = invariant_after(p, state, before, old_aux)
                    stats[i]["invariants"] &= valid
                    if not valid:
                        p.failure = "invariant"; p.finished = True
                    if p.failure == "incomplete" and p.steps >= horizon(teacher, p.n):
                        stats[i]["cutoff"] = True
                except Exception as error:
                    stats[i]["technical_error"] = f"{type(error).__name__}: {error}"
                    p.failure = "technical_error"; p.finished = True
        for (g, meta), p, stat in zip(part, programs, stats):
            proper = bool(p.bad_edges() == 0)
            palette = bool(((p.colors >= -1) & (p.colors < 4)).all())
            terminal = bool((p.colors >= 0).all()) and proper and palette
            row = dict(**meta, teacher=teacher, proper=proper, palette=palette,
                       invariants=bool(stat["invariants"]), terminated=bool(p.finished and not stat["cutoff"]),
                       no_technical_error=stat["technical_error"] is None, no_cutoff=not stat["cutoff"],
                       terminal_correct=terminal, failure=p.failure, technical_error=stat["technical_error"],
                       native_steps=p.steps, horizon=horizon(teacher, p.n), colors=p.colors.tolist(),
                       action_correct=stat["action_correct"], action_total=stat["action_total"])
            if not passed_run(row) and row["failure"] is None: row["failure"] = "incorrect_terminal"
            rows.append(row)
    return rows, time.perf_counter() - tick


@torch.inference_mode()
def teacher_state_fidelity(model, teacher, cases, limit=256):
    """Separate action/transition equality on teacher states, no optimizer access."""
    correct = total = transitions = same_transitions = 0
    phase_counts = defaultdict(lambda: [0, 0])
    model.eval()
    for g, meta in cases:
        p = ColoringProgram(g, teacher, meta["tape_seed"], max_rounds=horizon(teacher, len(g)))
        while not p.finished and total < limit:
            state = p.snapshot(); exact = teacher_choices(state, p.neighbors, teacher)
            roots = np.flatnonzero(state.permission)[:limit-total]
            pred = predictions(model, encode_roots(state, p.neighbors, roots), np.full(len(roots), state.phase))
            index = 1 if state.phase == LINIAL else (2 if state.phase == DETECT else 0)
            truth = (exact.color, exact.auxiliary, exact.mark)[index][roots]
            equal = pred[index] == truth
            correct += int(equal.sum()); total += len(roots)
            phase_counts[str(state.phase)][0] += int(equal.sum()); phase_counts[str(state.phase)][1] += len(roots)
            if len(roots) == int(state.permission.sum()):
                transitions += 1; same_transitions += int(equal.all())
            p.apply(state, exact)
        if total >= limit: break
    return dict(action_correct=correct, action_total=total, action_fidelity=correct/total if total else None,
                complete_transitions=transitions, identical_transitions=same_transitions,
                transition_fidelity=same_transitions/transitions if transitions else None,
                by_phase=dict(phase_counts))
