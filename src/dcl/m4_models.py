from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dcl.m4_data import M4_FEATURES
from dcl.staged_models import (
    neighbor_max,
    count_parameters,
)


def _intermediate_representations(
    x,
    participation_logit,
    candidate_logits,
    hard_messages,
):
    # Current state is known exactly.
    uncolored = (
        x[:, 0] > 0.5
    )

    if hard_messages:
        participation = (
            participation_logit > 0
        ).float()

        candidate_class = (
            candidate_logits
            .argmax(dim=1)
        )

        candidate = F.one_hot(
            candidate_class,
            num_classes=5,
        ).float()

    else:
        participation = torch.sigmoid(
            participation_logit
        )

        candidate = torch.softmax(
            candidate_logits,
            dim=1,
        )

    # Already-colored nodes cannot participate.
    participation = (
        participation
        * uncolored.float()
    )

    # Already-colored nodes have no candidate offer.
    no_candidate = torch.zeros_like(
        candidate
    )
    no_candidate[:, 0] = 1.0

    candidate = torch.where(
        uncolored[:, None],
        candidate,
        no_candidate,
    )

    return (
        participation,
        candidate,
    )


class M4A(nn.Module):
    """
    Complete oracle imitation.

    Architecture A:
        raw features
        -> MAX
        -> candidate

        predicted participate + candidate
        -> MAX
        -> commit
    """

    def __init__(
        self,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # color + activation_draw
        self.participation_head = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # own 7 + raw neighbor max 7
        self.candidate_head = nn.Sequential(
            nn.Linear(
                2 * M4_FEATURES,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                5,
            ),
        )

        # x(7) + participate(1) + candidate(5)
        phase2_dim = 13

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
        # ----- PARTICIPATION -----

        participation_logit = (
            self.participation_head(
                x[:, :6]
            )
            .squeeze(-1)
        )

        # ----- CANDIDATE OFFER -----

        neighbor_state = neighbor_max(
            x,
            edge_index,
        )

        candidate_input = torch.cat(
            [
                x,
                neighbor_state,
            ],
            dim=1,
        )

        candidate_logits = (
            self.candidate_head(
                candidate_input
            )
        )

        (
            participation_repr,
            candidate_repr,
        ) = _intermediate_representations(
            x=x,
            participation_logit=participation_logit,
            candidate_logits=candidate_logits,
            hard_messages=hard_messages,
        )

        # ----- COMMIT -----

        phase2 = torch.cat(
            [
                x,
                participation_repr[:, None],
                candidate_repr,
            ],
            dim=1,
        )

        neighbor_phase2 = neighbor_max(
            phase2,
            edge_index,
        )

        commit_input = torch.cat(
            [
                phase2,
                neighbor_phase2,
            ],
            dim=1,
        )

        commit_logit = (
            self.commit_head(
                commit_input
            )
            .squeeze(-1)
        )

        return (
            participation_logit,
            candidate_logits,
            commit_logit,
        )


class M4B(nn.Module):
    """
    Complete oracle imitation.

    Architecture B:
        learned message
        -> MAX
        -> candidate

        predicted participate + candidate
        -> learned message
        -> MAX
        -> commit
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

        # First communication:
        # learn what information about the state
        # should be sent to neighbors.
        self.message1 = nn.Sequential(
            nn.Linear(
                M4_FEATURES,
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
                M4_FEATURES
                + message_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                5,
            ),
        )

        phase2_dim = 13

        # Second communication:
        # learn how participation and candidate
        # should be communicated.
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
        # ----- PARTICIPATION -----

        participation_logit = (
            self.participation_head(
                x[:, :6]
            )
            .squeeze(-1)
        )

        # ----- CANDIDATE OFFER -----

        message1 = self.message1(
            x
        )

        neighbor_state = neighbor_max(
            message1,
            edge_index,
        )

        candidate_input = torch.cat(
            [
                x,
                neighbor_state,
            ],
            dim=1,
        )

        candidate_logits = (
            self.candidate_head(
                candidate_input
            )
        )

        (
            participation_repr,
            candidate_repr,
        ) = _intermediate_representations(
            x=x,
            participation_logit=participation_logit,
            candidate_logits=candidate_logits,
            hard_messages=hard_messages,
        )

        # ----- COMMIT -----

        phase2 = torch.cat(
            [
                x,
                participation_repr[:, None],
                candidate_repr,
            ],
            dim=1,
        )

        message2 = self.message2(
            phase2
        )

        neighbor_phase2 = neighbor_max(
            message2,
            edge_index,
        )

        commit_input = torch.cat(
            [
                phase2,
                neighbor_phase2,
            ],
            dim=1,
        )

        commit_logit = (
            self.commit_head(
                commit_input
            )
            .squeeze(-1)
        )

        return (
            participation_logit,
            candidate_logits,
            commit_logit,
        )


def make_m4_model(
    arch: str,
    hidden_dim: int = 64,
    message_dim: int = 32,
):
    arch = arch.upper()

    if arch == "A":
        return M4A(
            hidden_dim=hidden_dim,
        )

    if arch == "B":
        return M4B(
            hidden_dim=hidden_dim,
            message_dim=message_dim,
        )

    raise ValueError(
        f"Unknown M4 architecture: {arch}"
    )
