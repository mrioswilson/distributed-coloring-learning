import torch

from dcl.graphs import (
    connected_random_regular_graph,
)
from dcl.m4_data import (
    encode_m4_round,
)
from dcl.m4_models import (
    M4A,
    M4B,
)
from dcl.oracle import run_oracle


def test_m4_encoding():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=101,
    )

    result = run_oracle(
        G=G,
        seed=102,
        save_trace=True,
    )

    ex = encode_m4_round(
        G,
        result.trace[0],
    )

    assert ex.x.shape == (
        40,
        7,
    )

    assert ex.edge_index.shape == (
        2,
        120,
    )

    assert ex.y_participate.shape == (
        40,
    )

    assert ex.y_candidate_offer.shape == (
        40,
    )

    assert ex.y_commit.shape == (
        40,
    )


def test_m4_model_shapes():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=103,
    )

    result = run_oracle(
        G=G,
        seed=104,
        save_trace=True,
    )

    ex = encode_m4_round(
        G,
        result.trace[0],
    )

    x = torch.from_numpy(
        ex.x
    ).float()

    edge_index = torch.from_numpy(
        ex.edge_index
    ).long()

    for model in [
        M4A(),
        M4B(),
    ]:
        participate, candidate, commit = model(
            x,
            edge_index,
        )

        assert participate.shape == (
            40,
        )

        assert candidate.shape == (
            40,
            5,
        )

        assert commit.shape == (
            40,
        )
