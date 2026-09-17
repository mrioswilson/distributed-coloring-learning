from __future__ import annotations

import torch
from torch import nn


def neighbor_max_aggregate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    For every node v, compute the componentwise maximum
    of the features x_u of its immediate neighbors u.
    """

    source = edge_index[0]
    target = edge_index[1]

    # Message sent through every directed edge u -> v.
    messages = x[source]

    aggregated = torch.zeros_like(x)

    target_expanded = target[:, None].expand_as(messages)

    aggregated.scatter_reduce_(
        dim=0,
        index=target_expanded,
        src=messages,
        reduce="amax",
        include_self=False,
    )

    return aggregated


class CommitGNN(nn.Module):
    """
    One-round LOCAL GNN.

    Input:
        own node features
        +
        componentwise max of immediate-neighbor features

    Output:
        one logit per node:
        positive -> COMMIT
        negative -> WAIT
    """

    def __init__(
        self,
        num_features: int = 11,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(2 * num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        neighbor_info = neighbor_max_aggregate(
            x,
            edge_index,
        )

        local_info = torch.cat(
            [x, neighbor_info],
            dim=1,
        )

        logits = self.classifier(local_info)

        return logits.squeeze(-1)
