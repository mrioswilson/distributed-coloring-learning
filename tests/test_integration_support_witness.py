"""A full native prefix for the off-support witness, with explicit legal IDs."""
import networkx as nx
import numpy as np

from dcl.multicf.oracles import confirmation
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, KW, teacher_choices
from dcl.samplecomplexity.evaluation import invariant_after


def test_occupied_kw_query_has_a_full_exact_prefix_from_legal_ids():
    p = ColoringProgram(nx.star_graph(3), "BE_BASE_D3", 17)
    # Condition on one coherent legal ID assignment, not an asserted PRNG draw.
    p.ids = np.array([6, 3, 0, 1])
    p.order = np.argsort(p.ids)
    p.blocks = np.array_split(p.order, min(4, p.n))
    p.aux = p.ids.copy()
    while not (p.phase == KW and p.bound == 7):
        assert p.steps < 15
        s = p.snapshot()
        old = p.aux.copy()
        old_colors = p.colors.copy()
        p.apply(s, teacher_choices(s, p.neighbors, p.teacher))
        assert invariant_after(p, s, old_colors, old)
    s = p.snapshot()
    assert p.steps == 12 and s.target == 6
    assert s.aux.tolist() == [6, 3, 0, 1]
    arrays = encode_roots(s, p.neighbors, np.array([0]))
    exact = teacher_choices(s, p.neighbors, p.teacher)
    assert exact.color[0] == 3  # candidate index2, positive action code3
    participation = np.zeros((1, 4), bool)
    participation[0, 0] = True
    candidates = np.zeros((1, 4), int)
    candidates[0, 0] = 2
    assert confirmation(arrays, p.teacher, participation, candidates).tolist() == [1.]
    candidates[0, 0] = 3
    assert confirmation(arrays, p.teacher, participation, candidates).tolist() == [0.]
