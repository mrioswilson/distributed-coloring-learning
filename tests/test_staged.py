import numpy as np
import torch

from dcl.graphs import (
    connected_random_regular_graph,
)
from dcl.oracle import (
    UNCOLORED,
    run_oracle,
    sample_random_trial,
    sample_random_trial_details,
)
from dcl.staged_data import (
    encode_m2_round,
    encode_m3_round,
)
from dcl.staged_models import (
    M2A,
    M2B,
    M3A,
    M3B,
)


def test_random_trial_wrapper_is_identical():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=1,
    )

    colors = np.full(
        40,
        UNCOLORED,
        dtype=np.int64,
    )

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    participate, candidate = (
        sample_random_trial(
            G=G,
            colors=colors,
            num_colors=4,
            rng=rng1,
        )
    )

    details = (
        sample_random_trial_details(
            G=G,
            colors=colors,
            num_colors=4,
            rng=rng2,
        )
    )

    assert np.array_equal(
        participate,
        details.participate,
    )

    assert np.array_equal(
        candidate,
        details.candidate,
    )


def test_candidate_offer_exists_for_uncolored():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=2,
    )

    colors = np.full(
        40,
        UNCOLORED,
        dtype=np.int64,
    )

    sample = sample_random_trial_details(
        G=G,
        colors=colors,
        num_colors=4,
        rng=np.random.default_rng(456),
    )

    assert np.all(
        sample.candidate_offer
        != UNCOLORED
    )


def test_m2_m3_encodings():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=3,
    )

    result = run_oracle(
        G=G,
        seed=4,
        save_trace=True,
    )

    trace = result.trace[0]

    m2 = encode_m2_round(
        G,
        trace,
    )

    m3 = encode_m3_round(
        G,
        trace,
    )

    assert m2.x.shape == (
        40,
        11,
    )

    assert m3.x.shape == (
        40,
        7,
    )

    assert m2.edge_index.shape == (
        2,
        120,
    )

    assert m3.edge_index.shape == (
        2,
        120,
    )


def test_staged_model_shapes():
    G = connected_random_regular_graph(
        n=40,
        degree=3,
        seed=5,
    )

    result = run_oracle(
        G=G,
        seed=6,
        save_trace=True,
    )

    m2 = encode_m2_round(
        G,
        result.trace[0],
    )

    m3 = encode_m3_round(
        G,
        result.trace[0],
    )

    edge2 = torch.from_numpy(
        m2.edge_index
    ).long()

    edge3 = torch.from_numpy(
        m3.edge_index
    ).long()

    x2 = torch.from_numpy(
        m2.x
    ).float()

    x3 = torch.from_numpy(
        m3.x
    ).float()

    for model in [
        M2A(),
        M2B(),
    ]:
        participation, commit = model(
            x2,
            edge2,
        )

        assert participation.shape == (
            40,
        )

        assert commit.shape == (
            40,
        )

    for model in [
        M3A(),
        M3B(),
    ]:
        candidate, commit = model(
            x3,
            edge3,
        )

        assert candidate.shape == (
            40,
            5,
        )

        assert commit.shape == (
            40,
        )
