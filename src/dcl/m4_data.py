from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from dcl.dataset import EncodedState
from dcl.oracle import RoundTrace, UNCOLORED


# current color: 5
# activation draw: 1
# candidate draw: 1
M4_FEATURES = 7


@dataclass
class M4Example:
    x: np.ndarray
    edge_index: np.ndarray

    y_participate: np.ndarray
    y_candidate_offer: np.ndarray
    y_commit: np.ndarray

    decision_mask: np.ndarray


def _color_index(c: int) -> int:
    if c == UNCOLORED:
        return 0

    return c + 1


def _candidate_index(c: int) -> int:
    if c == UNCOLORED:
        return 0

    return c + 1


def _edge_index(
    G: nx.Graph,
) -> np.ndarray:
    edges = []

    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))

    return np.asarray(
        edges,
        dtype=np.int64,
    ).T


def encode_m4_state(
    G: nx.Graph,
    colors: np.ndarray,
    activation_draw: np.ndarray,
    candidate_draw: np.ndarray,
) -> EncodedState:
    """
    The model sees only:

        current color
        activation_draw
        candidate_draw

    It does NOT see:
        participate
        candidate_offer
        candidate
        commit
    """

    n = G.number_of_nodes()

    x = np.zeros(
        (n, M4_FEATURES),
        dtype=np.float32,
    )

    for v in range(n):
        color_idx = _color_index(
            int(colors[v])
        )

        x[v, color_idx] = 1.0

        x[v, 5] = float(
            activation_draw[v]
        )

        x[v, 6] = float(
            candidate_draw[v]
        )

    return EncodedState(
        x=x,
        edge_index=_edge_index(G),
    )


def encode_m4_round(
    G: nx.Graph,
    trace: RoundTrace,
) -> M4Example:
    if (
        trace.activation_draw is None
        or
        trace.candidate_draw is None
        or
        trace.candidate_offer is None
    ):
        raise ValueError(
            "Trace does not contain M4 random-tape data."
        )

    state = encode_m4_state(
        G=G,
        colors=trace.colors_before,
        activation_draw=trace.activation_draw,
        candidate_draw=trace.candidate_draw,
    )

    candidate_offer = np.asarray(
        [
            _candidate_index(int(c))
            for c in trace.candidate_offer
        ],
        dtype=np.int64,
    )

    return M4Example(
        x=state.x,
        edge_index=state.edge_index,

        y_participate=(
            trace.participate
            .astype(np.float32)
        ),

        y_candidate_offer=candidate_offer,

        y_commit=(
            trace.commit
            .astype(np.float32)
        ),

        decision_mask=(
            trace.colors_before
            == UNCOLORED
        ),
    )
