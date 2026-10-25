import networkx as nx
import numpy as np
import pytest
import torch

from dcl.degree.teachers import DegreeProgram, EXTENSIBLE, exact_choices, scalar_choices
from dcl.degree.graphs import graph_for, FAMILIES
from dcl.degree.adapter import FrozenAdapter
from dcl.multiteacher.programs import ColoringProgram, teacher_choices
from dcl.multiteacher.model import encode_roots
from dcl.samplecomplexity.model import QueryPolicy

torch.set_num_threads(1)


@pytest.mark.parametrize("teacher", EXTENSIBLE)
def test_historical_native_transitions(teacher):
    for g in (nx.path_graph(8), nx.star_graph(3), nx.complete_graph(4), nx.empty_graph(5),
              nx.random_regular_graph(3, 16, seed=4)):
        a, b = ColoringProgram(g, teacher, 100), DegreeProgram(g, teacher, 100, "native4")
        while not a.finished:
            x, y = a.snapshot(), b.snapshot()
            for key in ("colors", "permission", "draws", "preferences", "ids", "slots"):
                np.testing.assert_array_equal(getattr(x, key), getattr(y, key))
            left, right = teacher_choices(x, a.neighbors, teacher), exact_choices(b, y)
            for key in ("color", "auxiliary", "mark"):
                np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
            a.apply(x, left)
            b.apply(y, right)
            np.testing.assert_array_equal(a.colors, b.colors)
            assert a.finished == b.finished and a.failure == b.failure


@pytest.mark.parametrize("teacher", EXTENSIBLE)
@pytest.mark.parametrize("palette", ("global", "local"))
def test_scalar_differential_and_invariants(teacher, palette):
    for family in FAMILIES:
        g = graph_for(family, 24, 5, 44)
        p = DegreeProgram(g, teacher, 991, palette)
        while not p.finished:
            state = p.snapshot()
            a, b = exact_choices(p, state), scalar_choices(p, state)
            np.testing.assert_array_equal(a.color, b.color)
            np.testing.assert_array_equal(a.mark, b.mark)
            before = p.colors.copy()
            p.apply(state, a)
            assert np.all(p.colors < p.palette)
            if teacher != "SPECULATIVE_GM":
                assert p.bad_edges() == 0
                np.testing.assert_array_equal(p.colors[before >= 0], before[before >= 0])
            else:
                np.testing.assert_array_equal(p.colors[~state.permission], before[~state.permission])
        assert p.failure is None


@pytest.mark.parametrize("teacher", EXTENSIBLE)
@pytest.mark.parametrize("architecture", ("normalized", "deepsets", "transformer"))
def test_frozen_adapter_native_logits(teacher, architecture):
    torch.manual_seed(222)
    model = QueryPolicy(teacher, architecture).eval()
    adapter = FrozenAdapter(model)
    for g in (nx.path_graph(8), nx.empty_graph(5), nx.random_regular_graph(3, 16, seed=9)):
        p = DegreeProgram(g, teacher, 184, "native4")
        with torch.inference_mode():
            while not p.finished:
                state = p.snapshot()
                roots = np.flatnonzero(state.permission)
                arrays = tuple(torch.from_numpy(a) for a in encode_roots(state, p.neighbors, roots))
                original = model(*arrays, phase=state.phase)
                generalized = adapter(p, state)
                np.testing.assert_allclose(generalized.scores, original.scores.numpy(), atol=1e-5, rtol=2e-5)
                np.testing.assert_allclose(generalized.p_logits, original.state.p_logits[:, 0].numpy(), atol=1e-5, rtol=2e-5)
                np.testing.assert_allclose(generalized.commit_logits, original.commit.numpy(), atol=1e-5, rtol=2e-5)
                np.testing.assert_array_equal(generalized.candidates, original.candidates[:, 0].numpy())
                p.apply(state, exact_choices(p, state))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("delta,n", ((2, 64), (3, 64), (50, 64), (200, 256)))
def test_exact_requested_degree(family, delta, n):
    g = graph_for(family, n, delta, 411)
    assert len(g) == n and max(dict(g.degree()).values()) == delta
    if family == "tree":
        assert nx.is_tree(g)
    if family == "bipartite":
        assert nx.is_bipartite(g)


def test_be_is_not_arbitrarily_extended():
    with pytest.raises(ValueError):
        DegreeProgram(nx.complete_graph(5), "BE_BASE_D3", 0)


def test_local_palette_sender_can_own_receiver_external_color():
    # A high-degree neighbor's color outside a leaf's list must still be
    # represented as a fixed neighbor of a different color, not truncated away.
    g = nx.star_graph(10)
    p = DegreeProgram(g, "LUBY", 1, "local")
    p.colors[0] = 10
    state = p.snapshot()
    model = QueryPolicy("LUBY").eval()
    out = FrozenAdapter(model)(p, state)
    assert np.isfinite(out.p_logits).all()
    assert np.all(out.candidates < p.palette[out.roots])


@pytest.mark.parametrize("palette", ("global", "local"))
@pytest.mark.parametrize("delta", (3, 10, 50, 200))
def test_reuse_historical_high_degree_luby_reference(palette, delta):
    from dcl.universal.rules import teacher_offer, exact_commit_labels
    p = DegreeProgram(graph_for("isolates", delta + 9, delta, 911), "LUBY", 195, palette)
    while not p.finished:
        s = p.snapshot()
        proposal, _ = teacher_offer(s.colors, s.draws[:, 1], p.edges, p.palette)
        part = s.permission & (s.draws[:, 0] < .5)
        commit = exact_commit_labels(s.colors, p.edges, part, proposal)
        expected = np.where(commit, proposal + 1, 0)
        actual = exact_choices(p, s)
        np.testing.assert_array_equal(actual.color, expected)
        p.apply(s, actual)
    assert p.failure is None


@pytest.mark.parametrize("architecture", ("normalized", "deepsets", "transformer"))
@pytest.mark.parametrize("palette", ("global", "local"))
def test_sparse_high_degree_against_original_color_shared_core(architecture, palette):
    from dcl.localpalettes.adapter import LocalPaletteAdapter
    torch.manual_seed(100)
    model = QueryPolicy("LUBY", architecture).eval()
    p = DegreeProgram(graph_for("isolates", 32, 17, 319), "LUBY", 194, palette)
    with torch.inference_mode():
        # Include both pending and fixed nodes and heterogeneous local lists.
        for _ in range(3):
            s = p.snapshot()
            a = FrozenAdapter(model)(p, s)
            b = LocalPaletteAdapter(model.core)(torch.from_numpy(s.colors), torch.from_numpy(s.draws[:, 0]),
                 torch.from_numpy(s.draws[:, 1]), torch.from_numpy(p.edges), torch.from_numpy(p.palette))
            np.testing.assert_allclose(a.scores, b.candidate_logits.numpy()[a.roots], atol=1e-5, rtol=2e-5)
            np.testing.assert_allclose(a.commit_logits, b.commit_logit.numpy()[a.roots], atol=1e-5, rtol=2e-5)
            p.apply(s, exact_choices(p, s))
            if p.finished:
                break


def test_frozen_inference_preserves_weights_and_source_state():
    model = QueryPolicy("COLOR_TRIALS").eval()
    p = DegreeProgram(nx.star_graph(10), "COLOR_TRIALS", 391, "local")
    s = p.snapshot()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    before = {k: v.copy() for k, v in vars(s).items() if isinstance(v, np.ndarray)}
    FrozenAdapter(model)(p, s)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, weights[k], rtol=0, atol=0)
    for k, v in before.items():
        np.testing.assert_array_equal(getattr(s, k), v)


@pytest.mark.parametrize("teacher", EXTENSIBLE)
@pytest.mark.parametrize("palette", ("global", "local"))
@pytest.mark.parametrize("delta", (50, 200))
def test_independent_scalar_high_degree_and_speculative_invariants(teacher, palette, delta):
    from dcl.multiteacher.programs import DETECT, REPAIR
    p = DegreeProgram(graph_for("isolates", delta+17, delta, 901), teacher, 131, palette)
    while not p.finished:
        s = p.snapshot()
        a, b = exact_choices(p, s), scalar_choices(p, s)
        np.testing.assert_array_equal(a.color, b.color)
        np.testing.assert_array_equal(a.mark, b.mark)
        p.apply(s, a)
        assert np.all(p.colors < p.palette)
        if teacher == "SPECULATIVE_GM":
            u, v = p.edges
            bad = (p.colors[u] >= 0) & (p.colors[u] == p.colors[v])
            if p.phase != REPAIR:
                assert np.all(p.slots[u[bad]] == p.slots[v[bad]])
            if s.phase == DETECT and (p.phase == REPAIR or p.finished):
                assert np.all(p.marks[u[bad]] | p.marks[v[bad]])
        else:
            assert p.bad_edges() == 0
    assert p.failure is None


def test_ct_preference_gap_is_a_legal_exact_winner_at_all_degrees():
    for delta in (3, 4, 10, 50, 200):
        g = nx.disjoint_union(nx.path_graph(2), nx.star_graph(delta))
        p = DegreeProgram(g, "COLOR_TRIALS", 1, "local")
        s = p.snapshot()
        s.preferences[:] = 0
        s.preferences[0, 0], s.preferences[1, 0] = delta, delta-1
        a, b = exact_choices(p, s), scalar_choices(p, s)
        np.testing.assert_array_equal(a.color, b.color)
        assert a.color[0] == 1 and a.color[1] == 0
        assert p.palette[0] == p.palette[1] == 2


def test_counterfactual_query_only_changes_the_declared_head_query():
    torch.manual_seed(1)
    p = DegreeProgram(nx.star_graph(5), "COLOR_TRIALS", 192, "local")
    s = p.snapshot()
    adapter = FrozenAdapter(QueryPolicy("COLOR_TRIALS"))
    base = adapter(p, s)
    query = np.zeros(p.n, np.int64)
    forced = adapter(p, s, overrides=dict(query=query))
    np.testing.assert_array_equal(forced.candidates, 0)
    np.testing.assert_array_equal(base.scores, forced.scores)
    assert np.all(p.colors == -1)
