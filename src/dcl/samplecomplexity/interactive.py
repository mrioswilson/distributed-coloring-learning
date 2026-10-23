"""Admissibility is checked before any optional on-policy teacher query."""
import numpy as np
from dcl.multiteacher.programs import LINIAL, KW


def admissible(program, state):
    if np.any((state.colors < -1) | (state.colors >= 4)):
        return False
    if program.teacher == "BE_BASE_D3":
        return bool(np.all((state.aux >= 0) & (state.aux < state.bound)) and
                    all(state.aux[u] != state.aux[v] for u, v in program.graph.edges()))
    if program.teacher == "SPECULATIVE_GM":
        # Native speculative history/repair invariants require a separate proof.
        return False
    return all(state.colors[u] < 0 or state.colors[u] != state.colors[v] for u, v in program.graph.edges())
