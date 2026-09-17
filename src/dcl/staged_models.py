from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dcl.staged_data import (
    M2_FEATURES,
    M3_FEATURES,
)


def neighbor_max(
    z: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    source = edge_index[0]
    target = edge_index[1]

    messages = z[source]

    aggregated = torch.zeros_like(z)

    target_expanded = (
        target[:, None]
        .expand_as(messages)
    )

    aggregated.scatter_reduce_(
        dim=0,
        index=target_expanded,
        src=messages,
        reduce="amax",
        include_self=False,
    )

    return aggregated


def count_parameters(
    model: nn.Module,
) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


class M2A(nn.Module):
    """
    M2A:
        learned participation
        raw node representation
        -> neighbor MAX
        -> commit head
    """

    def __init__(
        self,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # Participation depends only on:
        # current color + own activation draw.
        self.participation_head = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        phase2_dim = (
            M2_FEATURES + 1
        )

        self.commit_head = nn.Sequential(
            nn.Linear(
                2 * phase2_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        x,
        edge_index,
        hard_messages: bool = False,
    ):
        participation_input = x[:, :6]

        participation_logit = (
            self.participation_head(
                participation_input
            )
            .squeeze(-1)
        )

        if hard_messages:
            participation_repr = (
                participation_logit > 0
            ).float()
        else:
            participation_repr = torch.sigmoid(
                participation_logit
            )

        phase2 = torch.cat(
            [
                x,
                participation_repr[:, None],
            ],
            dim=1,
        )

        neighbor_info = neighbor_max(
            phase2,
            edge_index,
        )

        local = torch.cat(
            [
                phase2,
                neighbor_info,
            ],
            dim=1,
        )

        commit_logit = (
            self.commit_head(local)
            .squeeze(-1)
        )

        return (
            participation_logit,
            commit_logit,
        )


class M2B(nn.Module):
    """
    M2B:
        learned participation
        -> learned message MLP
        -> neighbor MAX
        -> commit head
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        message_dim: int = 32,
    ):
        super().__init__()

        self.participation_head = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        phase2_dim = (
            M2_FEATURES + 1
        )

        self.message_net = nn.Sequential(
            nn.Linear(
                phase2_dim,
                message_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                message_dim,
                message_dim,
            ),
            nn.ReLU(),
        )

        self.commit_head = nn.Sequential(
            nn.Linear(
                phase2_dim
                + message_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        x,
        edge_index,
        hard_messages: bool = False,
    ):
        participation_logit = (
            self.participation_head(
                x[:, :6]
            )
            .squeeze(-1)
        )

        if hard_messages:
            participation_repr = (
                participation_logit > 0
            ).float()
        else:
            participation_repr = torch.sigmoid(
                participation_logit
            )

        phase2 = torch.cat(
            [
                x,
                participation_repr[:, None],
            ],
            dim=1,
        )

        messages = self.message_net(
            phase2
        )

        neighbor_info = neighbor_max(
            messages,
            edge_index,
        )

        local = torch.cat(
            [
                phase2,
                neighbor_info,
            ],
            dim=1,
        )

        commit_logit = (
            self.commit_head(local)
            .squeeze(-1)
        )

        return (
            participation_logit,
            commit_logit,
        )


class M3A(nn.Module):
    """
    M3A:

      phase 1:
        raw neighbor MAX
        -> candidate

      phase 2:
        predicted candidate
        -> raw neighbor MAX
        -> commit
    """

    def __init__(
        self,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.candidate_head = nn.Sequential(
            nn.Linear(
                2 * M3_FEATURES,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                5,
            ),
        )

        phase2_dim = (
            M3_FEATURES + 5
        )

        self.commit_head = nn.Sequential(
            nn.Linear(
                2 * phase2_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        x,
        edge_index,
        hard_messages: bool = False,
    ):
        neighbor_state = neighbor_max(
            x,
            edge_index,
        )

        candidate_local = torch.cat(
            [
                x,
                neighbor_state,
            ],
            dim=1,
        )

        candidate_logits = (
            self.candidate_head(
                candidate_local
            )
        )

        if hard_messages:
            cls = candidate_logits.argmax(
                dim=1
            )

            candidate_repr = F.one_hot(
                cls,
                num_classes=5,
            ).float()
        else:
            candidate_repr = torch.softmax(
                candidate_logits,
                dim=1,
            )

        phase2 = torch.cat(
            [
                x,
                candidate_repr,
            ],
            dim=1,
        )

        neighbor_candidate = neighbor_max(
            phase2,
            edge_index,
        )

        commit_local = torch.cat(
            [
                phase2,
                neighbor_candidate,
            ],
            dim=1,
        )

        commit_logit = (
            self.commit_head(
                commit_local
            )
            .squeeze(-1)
        )

        return (
            candidate_logits,
            commit_logit,
        )


class M3B(nn.Module):
    """
    M3B:

      phase 1:
        learned message
        -> MAX
        -> candidate

      phase 2:
        learned candidate message
        -> MAX
        -> commit
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        message_dim: int = 32,
    ):
        super().__init__()

        self.message1 = nn.Sequential(
            nn.Linear(
                M3_FEATURES,
                message_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                message_dim,
                message_dim,
            ),
            nn.ReLU(),
        )

        self.candidate_head = nn.Sequential(
            nn.Linear(
                M3_FEATURES
                + message_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                5,
            ),
        )

        phase2_dim = (
            M3_FEATURES + 5
        )

        self.message2 = nn.Sequential(
            nn.Linear(
                phase2_dim,
                message_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                message_dim,
                message_dim,
            ),
            nn.ReLU(),
        )

        self.commit_head = nn.Sequential(
            nn.Linear(
                phase2_dim
                + message_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        x,
        edge_index,
        hard_messages: bool = False,
    ):
        message1 = self.message1(x)

        neighbor_state = neighbor_max(
            message1,
            edge_index,
        )

        candidate_local = torch.cat(
            [
                x,
                neighbor_state,
            ],
            dim=1,
        )

        candidate_logits = (
            self.candidate_head(
                candidate_local
            )
        )

        if hard_messages:
            cls = candidate_logits.argmax(
                dim=1
            )

            candidate_repr = F.one_hot(
                cls,
                num_classes=5,
            ).float()
        else:
            candidate_repr = torch.softmax(
                candidate_logits,
                dim=1,
            )

        phase2 = torch.cat(
            [
                x,
                candidate_repr,
            ],
            dim=1,
        )

        message2 = self.message2(
            phase2
        )

        neighbor_candidate = neighbor_max(
            message2,
            edge_index,
        )

        commit_local = torch.cat(
            [
                phase2,
                neighbor_candidate,
            ],
            dim=1,
        )

        commit_logit = (
            self.commit_head(
                commit_local
            )
            .squeeze(-1)
        )

        return (
            candidate_logits,
            commit_logit,
        )


def make_model(
    task: str,
    arch: str,
    hidden_dim: int = 64,
    message_dim: int = 32,
):
    task = task.lower()
    arch = arch.upper()

    if task == "m2" and arch == "A":
        return M2A(
            hidden_dim=hidden_dim,
        )

    if task == "m2" and arch == "B":
        return M2B(
            hidden_dim=hidden_dim,
            message_dim=message_dim,
        )

    if task == "m3" and arch == "A":
        return M3A(
            hidden_dim=hidden_dim,
        )

    if task == "m3" and arch == "B":
        return M3B(
            hidden_dim=hidden_dim,
            message_dim=message_dim,
        )

    raise ValueError(
        f"Unknown task/architecture: "
        f"{task}/{arch}"
    )
