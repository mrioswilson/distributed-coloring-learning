import networkx as nx
import numpy as np
import pytest

from dcl.multiteacher.programs import ColoringProgram, TEACHERS, teacher_choices


@pytest.mark.parametrize("teacher", TEACHERS)
@pytest.mark.parametrize("graph", [nx.DiGraph([(0, 1)]), nx.MultiGraph([(0, 1), (0, 1)]), nx.Graph([(0, 0)])])
def test_native_program_rejects_graphs_outside_declared_simple_undirected_domain(teacher, graph):
    with pytest.raises(ValueError, match="simple undirected"):
        ColoringProgram(graph, teacher, 1)


def test_jp_round_cap_is_not_a_deterministic_termination_guarantee():
    p = ColoringProgram(nx.path_graph(129), "JP", 1, max_rounds=128)
    p.ids = np.arange(129)
    while not p.finished:
        state = p.snapshot()
        p.apply(state, teacher_choices(state, p.neighbors, "JP"))
    assert p.failure == "incomplete" and p.steps == 128
    assert (p.colors < 0).sum() == 1
