"""Universal Color GNN: one representation per (node, color), shared parameters.

The constructor takes no palette size. A forward pass receives only the local
protocol state (current colors, the two random draws), the edges and the
declared palette size of each node. It never receives available colors,
teacher proposals or commit labels; see docs/UCGNN_ARQUITECTURA.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import VARIANTS

NEG = -1e9


def mlp(inputs, hidden, outputs, final_relu=False, final_bias=True):
    layers = [nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs, bias=final_bias)]
    if final_relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def neighbor_max(z, edge_index, chunk=None):
    """Max over in-neighbors of non-negative rows; zeros for isolated nodes.

    ``z`` has shape [N, F]. Edges are processed in chunks of ``chunk`` edges to
    bound memory at O(chunk * F); the result does not depend on the chunking.
    """
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros_like(z)
    total = src.numel()
    step = total if not chunk else int(chunk)
    for start in range(0, total, max(step, 1)):
        s, d = src[start:start + step], dst[start:start + step]
        # Zero initialization and include_self=True equal include_self=False
        # because every message is a ReLU output (non-negative).
        out = out.scatter_reduce(0, d[:, None].expand(-1, z.shape[1]), z[s], reduce="amax", include_self=True)
    return out


@dataclass
class UOutput:
    participation_logit: torch.Tensor   # [N]
    candidate_logits: torch.Tensor      # [N, K]; NEG outside the declared palette
    commit_logit: torch.Tensor          # [N], for the hard proposals below
    participate: torch.Tensor           # [N] bool, hard
    candidate: torch.Tensor             # [N] long, -1 = no candidate
    availability: torch.Tensor          # [N, K] learned alpha (diagnostic)
    k_hat: torch.Tensor                 # [N]
    lo: torch.Tensor                    # [N, K]
    hi: torch.Tensor                    # [N, K]


class UniversalColorGNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, message_dim: int = 32, variant: str = "NORMALIZED"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}")
        self.variant = variant
        d = message_dim
        # Stage 1: per-color state message [uncolored(u), own(u,c)].
        self.message1 = mlp(2, d, d, final_relu=True)
        self.state1 = mlp(1 + d, hidden_dim, d, final_relu=True)
        self.availability = nn.Linear(d, 1)
        extra = 4 if variant == "NORMALIZED" else 5
        # No output bias: a constant shared by all colors cancels in the softmax.
        self.scorer = mlp(d + extra, hidden_dim, 1, final_bias=False)
        self.participation = mlp(2, 32, 1)
        # Stage 2: per-color proposal message [uncolored, own, participate, offer].
        self.message2 = mlp(4, d, d, final_relu=True)
        self.state2 = mlp(4 + d, hidden_dim, d, final_relu=True)
        self.commit_head = mlp(d + 3, hidden_dim, 1)

    @staticmethod
    def _slots(colors, palette):
        palette = palette.to(colors.device, torch.long)
        if bool((palette < 1).any()):
            raise ValueError("Palette sizes must be positive")
        if bool(((colors < -1) | (colors >= palette)).any()):
            raise ValueError("Colors must be -1 or inside the declared palette")
        k = int(palette.max())
        slots = torch.arange(k, device=colors.device)
        in_palette = slots[None, :] < palette[:, None]
        own = (colors[:, None] == slots[None, :]).float()
        uncolored = colors < 0
        return slots, in_palette, own, uncolored

    def stage1(self, colors, rp, rc, edge_index, palette, edge_chunk=None):
        slots, in_palette, own, uncolored = self._slots(colors, palette)
        n, k = own.shape
        u = uncolored.float()
        f1 = torch.stack((u[:, None].expand(n, k), own), dim=-1)
        m1 = self.message1(f1)
        d = m1.shape[-1]
        received = neighbor_max(m1.reshape(n, k * d), edge_index, edge_chunk).reshape(n, k, d)
        h1 = self.state1(torch.cat((u[:, None, None].expand(n, k, 1), received), dim=-1))
        alpha = torch.sigmoid(self.availability(h1).squeeze(-1)) * in_palette
        k_hat = alpha.sum(1)
        prefix = torch.cumsum(alpha, 1) - alpha
        denominator = k_hat.clamp_min(0.5)[:, None]
        lo, hi = prefix / denominator, (prefix + alpha) / denominator
        r = rc[:, None].expand(n, k)
        if self.variant == "NORMALIZED":
            extra = (alpha, lo, hi, r)
        else:
            extra = (alpha, prefix, prefix + alpha, k_hat[:, None].expand(n, k), r)
        scores = self.scorer(torch.cat((h1, torch.stack(extra, dim=-1)), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~in_palette, NEG)
        p_logit = self.participation(torch.stack((rp, u), dim=-1)).squeeze(-1)
        return p_logit, scores, alpha, k_hat, lo, hi

    def commit_logits(self, colors, edge_index, palette, participate, candidate, edge_chunk=None):
        """COMMIT pathway for externally supplied hard proposals (-1 = none)."""
        slots, in_palette, own, uncolored = self._slots(colors, palette)
        n, k = own.shape
        candidate = candidate.to(colors.device, torch.long)
        participate = participate.to(colors.device).bool() & uncolored
        candidate = torch.where(uncolored, candidate, torch.full_like(candidate, -1))
        if bool(((candidate < -1) | (candidate >= palette.to(colors.device))).any()):
            raise ValueError("Candidates must be -1 or inside the declared palette")
        offer = (candidate[:, None] == slots[None, :]).float()
        u, p = uncolored.float(), participate.float()
        f2 = torch.stack((u[:, None].expand(n, k), own, p[:, None].expand(n, k), offer), dim=-1)
        m2 = self.message2(f2)
        d = m2.shape[-1]
        received = neighbor_max(m2.reshape(n, k * d), edge_index, edge_chunk).reshape(n, k, d)
        z = self.state2(torch.cat((f2, received), dim=-1))
        # Read only the slot of the node's own candidate (zero if none).
        gathered = (offer[:, :, None] * z).sum(1)
        head_input = torch.cat((gathered, p[:, None], u[:, None], offer.sum(1, keepdim=True)), dim=-1)
        return self.commit_head(head_input).squeeze(-1)

    def forward(self, colors, rp, rc, edge_index, palette, edge_chunk=None):
        p_logit, scores, alpha, k_hat, lo, hi = self.stage1(colors, rp, rc, edge_index, palette, edge_chunk)
        uncolored = colors < 0
        participate = (p_logit > 0) & uncolored
        candidate = torch.where(uncolored, scores.argmax(1), torch.full_like(colors, -1))
        commit = self.commit_logits(colors, edge_index, palette, participate, candidate, edge_chunk)
        return UOutput(p_logit, scores, commit, participate, candidate, alpha, k_hat, lo, hi)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
