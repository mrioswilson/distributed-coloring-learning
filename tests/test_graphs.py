import networkx as nx
import pytest

from dcl.graphs import connected_random_regular_graph


def test_connected_cubic_graph():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=123,
    )

    assert G.number_of_nodes() == 40
    assert G.number_of_edges() == 60
    assert nx.is_connected(G)
    assert set(dict(G.degree()).values()) == {3}
    assert set(G.nodes()) == set(range(40))


def test_reproducible_seed():
    G1 = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=123,
    )

    G2 = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=123,
    )

    assert set(G1.edges()) == set(G2.edges())


def test_invalid_parity():
    with pytest.raises(ValueError):
        connected_random_regular_graph(
            n=41,
            degree=3,
            seed=0,
        )
