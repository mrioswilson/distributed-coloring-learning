"""Small legal executions that distinguish sufficient and necessary conditions."""
import networkx as nx
import numpy as np

from dcl.coauthorintegration.functional import kw_reference_messages
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, Choices, KW, teacher_choices
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.evaluation import invariant_after


def test_kw_omitted_eligible_write_immediately_breaks_reduced_bound():
    p = ColoringProgram(nx.star_graph(3), "BE_BASE_D3", 17)
    p.phase, p.bound = KW, 7
    p.aux = np.array([6, 3, 0, 1])
    s = p.snapshot()
    assert s.target == 6
    arrays = encode_roots(s, p.neighbors, np.arange(4))
    participation, candidate = kw_reference_messages(tensor_arrays(arrays))
    assert participation[:, 0].tolist() == [True, False, False, False]
    assert candidate[0, 0].item() == 2
    exact = teacher_choices(s, p.neighbors, p.teacher)
    assert exact.color.tolist() == [3, 0, 0, 0]
    old = p.aux.copy()
    p.apply(s, Choices(np.zeros(4, int), np.zeros(4, int), np.zeros(4, bool)))
    assert p.aux[0] == 6 and p.bound == 6
    assert not invariant_after(p, s, p.colors.copy(), old)


def test_gm_skipped_repair_can_be_resolved_by_a_later_endpoint():
    # Three one-vertex blocks assign simultaneously. Order gives a path low-mid-high.
    probe = ColoringProgram(nx.path_graph(3), "SPECULATIVE_GM", 23)
    low, mid, high = map(int, probe.order)
    g = nx.Graph(); g.add_nodes_from(range(3)); g.add_edges_from([(low,mid),(mid,high)])
    p = ColoringProgram(g, "SPECULATIVE_GM", 23)
    for _ in range(2):
        s = p.snapshot(); p.apply(s, teacher_choices(s, p.neighbors, p.teacher))
    assert p.repair_queue.tolist() == [low, mid]
    assert p.colors.tolist() == [0,0,0]
    s = p.snapshot()
    assert s.permission[low]
    p.apply(s, Choices(np.zeros(3,int), np.zeros(3,int), np.zeros(3,bool)))
    assert not p.finished and p.bad_edges() == 2
    s = p.snapshot(); p.apply(s, teacher_choices(s, p.neighbors, p.teacher))
    assert p.finished and p.failure is None and p.bad_edges() == 0
    assert p.colors[mid] == 1 and p.colors[low] == p.colors[high] == 0
