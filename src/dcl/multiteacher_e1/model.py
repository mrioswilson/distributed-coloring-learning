"""Rooted evaluation of the original E1 modules, plus small native adapters.

Luby, Johansson and sequential greedy use precisely the 16035-parameter
UniversalColorGNN. Native schedules only determine writable vertices. Other
teachers retain all E1 modules and add context residuals, still shared across
nodes/colors and aggregated with MAX. ColorTrials reads its candidate after
the second communication: neighbors must first learn their availability.
No exact availability, winning flag or teacher choice is a model input.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dcl.multiteacher.programs import ASSIGN, DETECT, KW, LINIAL, MAX_FIELD, REPAIR
from dcl.universal.model import NEG, UniversalColorGNN, mlp

CANONICAL = ("LUBY", "JOHANSSON", "GREEDY")


def binary_table(module, width):
    """Evaluate a message MLP once per binary input pattern (same function)."""
    codes = torch.arange(2 ** width)
    powers = 2 ** torch.arange(width - 1, -1, -1)
    inputs = ((codes[:, None] // powers) % 2).float()
    return module(inputs)


@dataclass
class State:
    centers: torch.Tensor
    adjacent: torch.Tensor
    relations: torch.Tensor
    masks: torch.Tensor
    u: torch.Tensor
    own: torch.Tensor
    valid: torch.Tensor
    local: torch.Tensor
    h1: torch.Tensor
    alpha: torch.Tensor
    p_logits: torch.Tensor
    scores: torch.Tensor
    phase: int


@dataclass
class Output:
    state: State
    p: torch.Tensor
    candidates: torch.Tensor
    scores: torch.Tensor
    commit: torch.Tensor


class RootedE1(nn.Module):
    def __init__(self, teacher, hidden=64, message=32):
        super().__init__()
        self.teacher = teacher
        self.core = UniversalColorGNN(hidden, message, "NORMALIZED")
        self.native = teacher not in CANONICAL
        if self.native:
            self.context_message1 = mlp(8, 16, message, final_relu=True)
            self.context_state1 = nn.Linear(8, message)
            self.context_participation = mlp(message + 8, 32, 1)
            self.context_message2 = mlp(12, 16, message, final_relu=True)
            self.context_state2 = nn.Linear(8, message)

    def _scores(self, h, valid, rc):
        alpha = torch.sigmoid(self.core.availability(h).squeeze(-1)) * valid
        prefix = alpha.cumsum(-1) - alpha
        denominator = alpha.sum(-1, keepdim=True).clamp_min(.5)
        extra = torch.stack((alpha, prefix / denominator, (prefix + alpha) / denominator,
                             rc[..., None].expand_as(alpha)), dim=-1)
        scores = self.core.scorer(torch.cat((h, extra), dim=-1)).squeeze(-1)
        return scores.masked_fill(~valid, NEG), alpha

    def first(self, arrays, phase):
        x, y, rel, masks = arrays
        b, centers, degree, _ = y.shape
        k = int((x[:, 0, 15] * MAX_FIELD).round().max()) if phase == LINIAL else 4
        center_valid = torch.cat((torch.ones((b, 1), dtype=torch.bool), masks[:, 0]), dim=1)
        u, neighbor_u = x[..., 0], y[..., 0]
        if self.teacher == "SPECULATIVE_GM":
            u, neighbor_u = torch.maximum(u, x[..., 7]), torch.maximum(neighbor_u, y[..., 7])
        if self.teacher == "BE_BASE_D3":
            u = center_valid.float()
            neighbor_u = masks.float()
        if phase in (LINIAL, KW):
            own = torch.zeros((b, centers, k), dtype=x.dtype)
            occupied = (rel[..., 2:2 + k] if phase == LINIAL else rel[..., -4:])
        else:
            own, occupied = x[..., 1:5], y[..., 1:5]
        valid = torch.ones((b, centers, k), dtype=torch.bool)
        if phase == LINIAL:
            valid = torch.arange(k)[None, None] < (x[..., 15] * MAX_FIELD).round()[..., None]
            valid |= ~center_valid[..., None]  # padding is ignored in aggregation
        message1 = binary_table(self.core.message1, 2)
        codes = neighbor_u.long()[..., None] * 2 + occupied.long()
        local = torch.cat((x[..., 8:14], x[..., 7:8],
                           x[..., 18:19] if phase == KW else torch.zeros_like(x[..., 18:19])), dim=-1)
        if self.native:
            messages = message1[codes]
            difference = y[..., 5] - x[..., None, 5]
            if self.teacher not in ("JP", "SPECULATIVE_GM"):
                difference = torch.zeros_like(difference)
            same_slot = rel[..., 1] if self.teacher == "SPECULATIVE_GM" else torch.zeros_like(difference)
            edge_context = torch.cat((x[..., None, 8:14].expand(b, centers, degree, 6),
                                      difference[..., None], same_slot[..., None]), dim=-1)
            messages = messages + self.context_message1(edge_context)[..., None, :]
            received = (messages * masks[..., None, None]).amax(2)
            h1 = self.core.state1(torch.cat((u[..., None, None].expand(b, centers, k, 1), received), dim=-1))
            h1 = torch.relu(h1 + self.context_state1(local)[..., None, :])
        else:
            # A MAX of messages from four binary categories has only 16
            # possible subsets. Evaluate state1 once per (u, subset), then
            # gather; this changes neither inputs nor the learned function.
            present = ((codes[..., None] == torch.arange(4)) & masks[..., None, None]).any(2)
            subset = (present.long() * (2 ** torch.arange(4))).sum(-1)
            members = ((torch.arange(16)[:, None] // (2 ** torch.arange(4))) % 2).float()
            received_table = (message1[None] * members[..., None]).amax(1)
            states = torch.arange(32)
            h_table = self.core.state1(torch.cat(((states // 16).float()[:, None],
                                                  received_table[states % 16]), dim=-1))
            h1 = h_table[u.long()[..., None] * 16 + subset]
        scores, alpha = self._scores(h1, valid, x[..., 33])
        p_logits = self.core.participation(torch.stack((x[..., 32], u), dim=-1)).squeeze(-1)
        if self.native:
            p_logits = p_logits + self.context_participation(torch.cat((h1.mean(2), local), dim=-1)).squeeze(-1)
        return State(x, y, rel, masks, u, own, valid, local, h1, alpha, p_logits, scores, phase)

    def proposals(self, state):
        p = (state.p_logits > 0) & (state.centers[..., 7] > .5)
        candidates = state.scores.argmax(-1)
        return p, candidates

    def second(self, state, p, candidates):
        x, u, own = state.centers, state.u, state.own
        b, centers, k = own.shape
        slots = torch.arange(k)
        offer = (candidates[..., None] == slots).float() * (u > .5)[..., None]
        # ColorTrials broadcasts preferences/learned availability for every color,
        # rather than a single provisional offer. DETECT queries the fixed color.
        if self.teacher == "COLOR_TRIALS":
            offer = torch.zeros_like(offer)
        f2 = torch.stack((u[..., None].expand_as(own), own,
                          p.float()[..., None].expand_as(own), offer), dim=-1)
        neighbor_f2 = f2[:, 1:]
        if state.phase == KW:
            # KW options are relative to the receiver's old auxiliary color.
            # Reuse the same raw equality input as stage1 in COMMIT's own-slot
            # channel; never replace it with an exact availability decision.
            neighbor_f2 = neighbor_f2.clone()
            neighbor_f2[..., 1] = state.relations[:, 0, :, -4:]
        codes = (neighbor_f2.long() * torch.tensor([8, 4, 2, 1])).sum(-1)
        messages = binary_table(self.core.message2, 4)[codes]
        if self.native:
            degree = centers - 1
            phase = x[:, :1, 8:14].expand(b, degree, 6)[..., None, :].expand(b, degree, k, 6)
            difference = x[:, 1:, 5] - x[:, :1, 5]
            same_slot = state.relations[:, 0, :, 1]
            if self.teacher not in ("JP", "SPECULATIVE_GM"):
                difference = torch.zeros_like(difference)
            if self.teacher != "SPECULATIVE_GM":
                same_slot = torch.zeros_like(same_slot)
            pair = torch.stack((difference, same_slot), dim=-1)[..., None, :].expand(b, degree, k, 2)
            if self.teacher == "COLOR_TRIALS":
                color_context = torch.stack((x[:, 1:, 34:38], x[:, :1, 34:38].expand(b, degree, k),
                                              state.alpha[:, 1:], state.alpha[:, :1].expand(b, degree, k)), dim=-1)
            else:
                color_context = torch.zeros((b, degree, k, 4), dtype=x.dtype)
            messages = messages + self.context_message2(torch.cat((phase, pair, color_context), dim=-1))
        received = (messages * state.masks[:, 0, :, None, None]).amax(1)
        z = self.core.state2(torch.cat((f2[:, 0], received), dim=-1))
        if self.native:
            z = torch.relu(z + self.context_state2(state.local[:, 0])[:, None, :])
        return z

    def read_commit(self, state, z, p, query):
        b, k, _ = z.shape
        offer = (query[:, None] == torch.arange(k)).float()
        gathered = (offer[..., None] * z).sum(1)
        inputs = torch.cat((gathered, p[:, :1].float(), state.u[:, :1], offer.sum(1, keepdim=True)), dim=-1)
        return self.core.commit_head(inputs).squeeze(-1)

    def commit_for(self, state, p, candidates):
        z = self.second(state, p, candidates)
        return self.read_commit(state, z, p, candidates[:, 0])

    def forward(self, *arrays, phase=0):
        state = self.first(arrays, phase)
        p, candidates = self.proposals(state)
        if phase == LINIAL:
            return Output(state, p, candidates, state.scores[:, 0], state.p_logits[:, 0] * 0)
        if phase == DETECT:
            candidates = state.own.argmax(-1)
            p = state.centers[..., 7] > .5
        z = self.second(state, p, candidates)
        scores = state.scores[:, 0]
        if self.teacher == "COLOR_TRIALS":
            scores, _ = self._scores(z, state.valid[:, 0], state.centers[:, 0, 33])
            candidates = torch.cat((scores.argmax(-1)[:, None], candidates[:, 1:]), dim=1)
        commit = self.read_commit(state, z, p, candidates[:, 0])
        return Output(state, p, candidates, scores, commit)


def parameters(model):
    return {"core": sum(p.numel() for p in model.core.parameters()),
            "total": sum(p.numel() for p in model.parameters())}


def native_prediction(output):
    """Convert hard learned decisions to the existing interpreter's alphabet."""
    state = output.state
    b = len(state.centers)
    color = torch.zeros(b, dtype=torch.long)
    auxiliary = torch.zeros_like(color)
    mark = torch.zeros(b, dtype=torch.bool)
    if state.phase == LINIAL:
        point = output.candidates[:, 0]
        # Polynomial arithmetic is a declared local representation adapter.
        value = (state.centers[:, 0, 19:32] * (MAX_FIELD - 1)).round().long()
        auxiliary = point * MAX_FIELD + value[torch.arange(b), point]
    elif state.phase == DETECT:
        mark = output.commit > 0
    else:
        write = output.p[:, 0] & (output.commit > 0)
        color = torch.where(write, output.candidates[:, 0] + 1, color)
    return color, auxiliary, mark
