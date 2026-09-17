import networkx as nx
import numpy as np

from dcl.graphs import connected_random_regular_graph
from dcl.oracle import (
    UNCOLORED,
    available_colors,
    compute_commit_mask,
    is_proper_coloring,
    run_oracle,
)


def test_available_colors():
    G = nx.path_graph(3)

    colors = np.array(
        [0, UNCOLORED, 2],
        dtype=np.int64,
    )

    palette = available_colors(
        G=G,
        colors=colors,
        num_colors=4,
        v=1,
    )

    assert set(palette) == {1, 3}


def test_equal_candidates_collide():
    G = nx.path_graph(2)

    colors = np.array(
        [UNCOLORED, UNCOLORED],
        dtype=np.int64,
    )

    participate = np.array([True, True])
    candidate = np.array([2, 2])

    commit = compute_commit_mask(
        G=G,
        colors=colors,
        participate=participate,
        candidate=candidate,
    )

    assert commit.tolist() == [False, False]


def test_different_candidates_commit():
    G = nx.path_graph(2)

    colors = np.array(
        [UNCOLORED, UNCOLORED],
        dtype=np.int64,
    )

    participate = np.array([True, True])
    candidate = np.array([1, 3])

    commit = compute_commit_mask(
        G=G,
        colors=colors,
        participate=participate,
        candidate=candidate,
    )

    assert commit.tolist() == [True, True]


def test_fixed_neighbor_blocks_candidate():
    G = nx.path_graph(2)

    colors = np.array(
        [1, UNCOLORED],
        dtype=np.int64,
    )

    participate = np.array([False, True])
    candidate = np.array([UNCOLORED, 1])

    commit = compute_commit_mask(
        G=G,
        colors=colors,
        participate=participate,
        candidate=candidate,
    )

    assert commit.tolist() == [False, False]


def test_oracle_solves_random_cubic_graphs():
    for i in range(20):
        G = connected_random_regular_graph(
            n=40,
            degree=3,
            seed=1000 + i,
        )

        result = run_oracle(
            G=G,
            seed=2000 + i,
            num_colors=4,
            max_rounds=1000,
            save_trace=False,
        )

        assert result.solved
        assert is_proper_coloring(G, result.colors)
