from __future__ import annotations

import networkx as nx


def connected_random_regular_graph(
    n: int,
    degree: int = 3,
    seed: int | None = None,
    max_tries: int = 100,
) -> nx.Graph:
    """
    Generate a connected random d-regular simple graph.

    Parameters
    ----------
    n:
        Number of vertices.
    degree:
        Regular degree d.
    seed:
        Base seed used by NetworkX.
    max_tries:
        Maximum number of graph samples before failing.
    """

    if n <= 0:
        raise ValueError("n must be positive")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    if degree >= n:
        raise ValueError("degree must be smaller than n")

    if n * degree % 2 != 0:
        raise ValueError("n * degree must be even")

    for attempt in range(max_tries):
        graph_seed = None if seed is None else seed + attempt

        G = nx.random_regular_graph(
            d=degree,
            n=n,
            seed=graph_seed,
        )

        if nx.is_connected(G):
            return nx.convert_node_labels_to_integers(
                G,
                first_label=0,
                ordering="default",
            )

    raise RuntimeError(
        f"Could not generate a connected {degree}-regular "
        f"graph on {n} vertices after {max_tries} attempts"
    )
