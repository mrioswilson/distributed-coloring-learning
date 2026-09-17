from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from dcl.dataset import EncodedState
from dcl.oracle import (
    RoundTrace,
    UNCOLORED,
)


M2_FEATURES = 11
M3_FEATURES = 7


@dataclass
class M2Example:
    x: np.ndarray
    edge_index: np.ndarray
    y_participate: np.ndarray
    y_commit: np.ndarray
    decision_mask: np.ndarray


@dataclass
class M3Example:
    x: np.ndarray
    edge_index: np.ndarray
    y_candidate: np.ndarray
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


def encode_m2_state(
    G: nx.Graph,
    colors: np.ndarray,
    activation_draw: np.ndarray,
    candidate_offer: np.ndarray,
) -> EncodedState:
    """
    M2 sees:

      current color:      5 dims
      activation draw:    1 dim
      candidate offer:    5 dims

    It does NOT see participate.
    """
    n = G.number_of_nodes()

    x = np.zeros(
        (n, M2_FEATURES),
        dtype=np.float32,
    )

    for v in range(n):
        x[
            v,
            _color_index(
                int(colors[v])
            ),
        ] = 1.0

        x[v, 5] = float(
            activation_draw[v]
        )

        offer_idx = _candidate_index(
            int(candidate_offer[v])
        )

        x[v, 6 + offer_idx] = 1.0

    return EncodedState(
        x=x,
        edge_index=_edge_index(G),
    )


def encode_m2_round(
    G: nx.Graph,
    trace: RoundTrace,
) -> M2Example:
    if (
        trace.activation_draw is None
        or
        trace.candidate_offer is None
    ):
        raise ValueError(
            "Trace does not contain M2 random-tape data."
        )

    state = encode_m2_state(
        G=G,
        colors=trace.colors_before,
        activation_draw=trace.activation_draw,
        candidate_offer=trace.candidate_offer,
    )

    return M2Example(
        x=state.x,
        edge_index=state.edge_index,
        y_participate=(
            trace.participate
            .astype(np.float32)
        ),
        y_commit=(
            trace.commit
            .astype(np.float32)
        ),
        decision_mask=(
            trace.colors_before
            == UNCOLORED
        ),
    )


def encode_m3_state(
    G: nx.Graph,
    colors: np.ndarray,
    participate: np.ndarray,
    candidate_draw: np.ndarray,
) -> EncodedState:
    """
    M3 sees:

      current color:      5 dims
      participate:        1 dim
      candidate draw:     1 dim

    It does NOT see candidate.
    """
    n = G.number_of_nodes()

    x = np.zeros(
        (n, M3_FEATURES),
        dtype=np.float32,
    )

    for v in range(n):
        x[
            v,
            _color_index(
                int(colors[v])
            ),
        ] = 1.0

        x[v, 5] = float(
            participate[v]
        )

        x[v, 6] = float(
            candidate_draw[v]
        )

    return EncodedState(
        x=x,
        edge_index=_edge_index(G),
    )


def encode_m3_round(
    G: nx.Graph,
    trace: RoundTrace,
) -> M3Example:
    if trace.candidate_draw is None:
        raise ValueError(
            "Trace does not contain M3 random-tape data."
        )

    state = encode_m3_state(
        G=G,
        colors=trace.colors_before,
        participate=trace.participate,
        candidate_draw=trace.candidate_draw,
    )

    y_candidate = np.asarray(
        [
            _candidate_index(int(c))
            for c in trace.candidate
        ],
        dtype=np.int64,
    )

    return M3Example(
        x=state.x,
        edge_index=state.edge_index,
        y_candidate=y_candidate,
        y_commit=(
            trace.commit
            .astype(np.float32)
        ),
        decision_mask=(
            trace.colors_before
            == UNCOLORED
        ),
    )
