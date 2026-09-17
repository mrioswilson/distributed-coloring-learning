import torch

from dcl.model import (
    CommitGNN,
    neighbor_max_aggregate,
)


def test_neighbor_max_aggregate():
    # Graph:
    #
    # 0 -- 1 -- 2
    #
    x = torch.tensor(
        [
            [1.0, 0.0],  # node 0
            [0.0, 1.0],  # node 1
            [1.0, 1.0],  # node 2
        ]
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],  # source
            [1, 0, 2, 1],  # target
        ]
    )

    result = neighbor_max_aggregate(
        x,
        edge_index,
    )

    expected = torch.tensor(
        [
            [0.0, 1.0],  # node 0 receives node 1
            [1.0, 1.0],  # node 1 receives nodes 0 and 2
            [0.0, 1.0],  # node 2 receives node 1
        ]
    )

    assert torch.equal(result, expected)


def test_commit_gnn_output_shape():
    model = CommitGNN(
        num_features=11,
        hidden_dim=32,
    )

    x = torch.randn(40, 11)

    # Just a valid collection of directed edges.
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ]
    )

    logits = model(
        x,
        edge_index,
    )

    assert logits.shape == (40,)
