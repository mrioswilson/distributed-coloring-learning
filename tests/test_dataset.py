import numpy as np

from dcl.dataset import encode_round
from dcl.graphs import connected_random_regular_graph
from dcl.oracle import run_oracle


def test_encode_round_shapes():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=123,
    )

    result = run_oracle(
        G=G,
        seed=456,
        num_colors=4,
        save_trace=True,
    )

    example = encode_round(
        G,
        result.trace[0],
    )

    assert example.x.shape == (40, 11)

    # 60 undirected edges -> 120 directed communication arcs.
    assert example.edge_index.shape == (2, 120)

    assert example.y.shape == (40,)
    assert example.decision_mask.shape == (40,)

    assert set(np.unique(example.y)).issubset(
        {0.0, 1.0}
    )


def test_edge_index_contains_both_directions():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=123,
    )

    result = run_oracle(
        G=G,
        seed=456,
        num_colors=4,
        save_trace=True,
    )

    example = encode_round(
        G,
        result.trace[0],
    )

    arcs = set(
        map(tuple, example.edge_index.T.tolist())
    )

    for u, v in G.edges():
        assert (u, v) in arcs
        assert (v, u) in arcs


def test_decision_mask_is_participation():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=123,
    )

    result = run_oracle(
        G=G,
        seed=456,
        num_colors=4,
        save_trace=True,
    )

    trace = result.trace[0]
    example = encode_round(G, trace)

    assert np.array_equal(
        example.decision_mask,
        trace.participate,
    )
