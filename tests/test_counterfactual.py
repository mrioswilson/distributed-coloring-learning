import networkx as nx
import numpy as np
import torch

from dcl.counterfactual import (
    commit_logits_for_proposals,
    exact_commit_labels,
    make_counterfactual_proposals,
)
from dcl.m4_data import encode_m4_state
from dcl.m4_models import make_m4_model
from dcl.oracle import compute_commit_mask


def test_exact_labels_match_oracle():
    G = nx.path_graph(4)
    colors = np.array([2, -1, -1, 1])

    state = encode_m4_state(
        G, colors, np.zeros(4), np.zeros(4)
    )

    participate = np.array(
        [False, True, True, False]
    )

    for candidates in (
        np.array([0, 3, 3, 0]),
        np.array([0, 4, 1, 0]),
        np.array([0, 1, 3, 0]),
    ):
        expected = compute_commit_mask(
            G, colors, participate, candidates - 1
        )

        actual = exact_commit_labels(
            state.x,
            state.edge_index,
            participate,
            candidates,
        )

        np.testing.assert_array_equal(
            actual, expected
        )


def test_counterfactual_proposals():
    G = nx.path_graph(4)
    colors = np.array([2, -1, -1, 1])

    state = encode_m4_state(
        G, colors, np.zeros(4), np.zeros(4)
    )

    p, c = make_counterfactual_proposals(
        state.x,
        state.edge_index,
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 2, 0]),
        np.random.default_rng(11),
    )

    assert p.shape == (4,)
    assert c.shape == (4,)
    assert np.all((c >= 0) & (c <= 4))

    assert not p[0] and not p[3]
    assert c[0] == 0 and c[3] == 0


def test_original_forward_and_all_gradients():
    torch.manual_seed(3)

    G = nx.path_graph(4)
    colors = np.array([2, -1, -1, 1])

    state = encode_m4_state(
        G,
        colors,
        np.full(4, 0.2),
        np.full(4, 0.2),
    )

    model = make_m4_model(
        "B", hidden_dim=8, message_dim=4
    )

    x = torch.tensor(state.x)
    edges = torch.tensor(state.edge_index)

    p, c, original_commit = model(
        x, edges, hard_messages=True
    )

    forced_commit = commit_logits_for_proposals(
        model,
        x,
        edges,
        p > 0,
        c.argmax(dim=1),
    )

    torch.testing.assert_close(
        original_commit, forced_commit
    )

    p_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            p[1:3], torch.tensor([1.0, 1.0])
        )
    )

    c_loss = torch.nn.functional.cross_entropy(
        c[1:3], torch.tensor([1, 2])
    )

    cf_logits = commit_logits_for_proposals(
        model,
        x,
        edges,
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([0, 3, 3, 0]),
    )

    cf_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            cf_logits[1:3],
            torch.tensor([0.0, 0.0]),
        )
    )

    (p_loss + c_loss + cf_loss).backward()

    for name in (
        "participation_head",
        "candidate_head",
        "message1",
        "message2",
        "commit_head",
    ):
        assert any(
            parameter.grad is not None
            for parameter in getattr(
                model, name
            ).parameters()
        ), name
