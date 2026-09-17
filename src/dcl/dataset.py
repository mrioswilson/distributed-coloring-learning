from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from dcl.oracle import UNCOLORED, RoundTrace


NUM_COLORS = 4
NUM_COLOR_STATES = NUM_COLORS + 1
NUM_NODE_FEATURES = 11


@dataclass
class EncodedState:
    x: np.ndarray
    edge_index: np.ndarray


@dataclass
class RoundExample:
    x: np.ndarray
    edge_index: np.ndarray
    y: np.ndarray
    decision_mask: np.ndarray


def color_to_index(color: int) -> int:
    if color == UNCOLORED:
        return 0

    return color + 1


def encode_state(
    G: nx.Graph,
    colors: np.ndarray,
    participate: np.ndarray,
    candidate: np.ndarray,
) -> EncodedState:
    n = G.number_of_nodes()

    x = np.zeros(
        (n, NUM_NODE_FEATURES),
        dtype=np.float32,
    )

    for v in range(n):
        # Estado de color actual: UNCOLORED, 0, 1, 2, 3
        color_idx = color_to_index(int(colors[v]))
        x[v, color_idx] = 1.0

        # Participación
        x[v, 5] = float(participate[v])

        # Candidato: NO_CANDIDATE, 0, 1, 2, 3
        c = int(candidate[v])

        if c == UNCOLORED:
            candidate_idx = 0
        else:
            candidate_idx = c + 1

        x[v, 6 + candidate_idx] = 1.0

    edges = []

    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))

    edge_index = np.asarray(
        edges,
        dtype=np.int64,
    ).T

    return EncodedState(
        x=x,
        edge_index=edge_index,
    )


def encode_round(
    G: nx.Graph,
    trace: RoundTrace,
) -> RoundExample:
    state = encode_state(
        G=G,
        colors=trace.colors_before,
        participate=trace.participate,
        candidate=trace.candidate,
    )

    return RoundExample(
        x=state.x,
        edge_index=state.edge_index,
        y=trace.commit.astype(np.float32),
        decision_mask=trace.participate.astype(bool),
    )
