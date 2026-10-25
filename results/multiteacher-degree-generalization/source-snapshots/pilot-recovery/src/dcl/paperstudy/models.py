"""Modular interventions. No oracle operation is reachable from these models.

The historical UniversalColorGNN remains unchanged. The isolated endpoint
intervention changes only the two endpoint features, not any tensor shape.
All palette reductions are local to a node, never over graph nodes.
"""
from __future__ import annotations

import torch
from torch import nn

from dcl.universal.model import NEG, UniversalColorGNN, mlp, neighbor_max

PRIMARY = ("normalized", "raw_endpoints", "sum", "mean", "deepsets", "transformer")


def neighbor_reduce(z, edge_index, reduction="max", chunk=None):
    if reduction == "max":
        return neighbor_max(z, edge_index, chunk)
    if reduction not in ("sum", "mean"):
        raise ValueError(reduction)
    src, dst = edge_index
    result = torch.zeros_like(z)
    step = max(1, int(chunk or src.numel()))
    for start in range(0, src.numel(), step):
        s, d = src[start:start + step], dst[start:start + step]
        result = result.index_add(0, d, z[s])
    if reduction == "mean":
        degree = torch.bincount(dst, minlength=z.shape[0]).clamp_min(1)
        result = result / degree[:, None]
    return result


class MechanismUCGNN(UniversalColorGNN):
    def __init__(self, treatment="normalized", hidden_dim=64, message_dim=32):
        super().__init__(hidden_dim, message_dim, "NORMALIZED")
        if treatment not in PRIMARY + ("own_only", "cyclic_teacher"):
            raise ValueError(treatment)
        self.treatment = treatment
        self.reduction = treatment if treatment in ("sum", "mean") else "max"
        if treatment == "deepsets":
            # Replaces the prefix scorer; the learned availability keeps a
            # gradient path through alpha. Position is functional, never capped.
            del self.scorer
            self.palette_phi = mlp(message_dim + 3, 16, 16, final_relu=True)
            self.scorer = mlp(message_dim + 4 + 16, hidden_dim, 1, final_bias=False)
        elif treatment == "transformer":
            del self.scorer
            self.palette_project = nn.Linear(message_dim + 4, 16)
            self.palette_attention = nn.TransformerEncoderLayer(
                d_model=16, nhead=2, dim_feedforward=32, dropout=0.0,
                activation="relu", batch_first=True, norm_first=False)
            self.scorer = mlp(16, 32, 1, final_bias=False)

    def palette_scores(self, h, alpha, rc, palette, in_palette):
        n, k = alpha.shape
        r = rc[:, None].expand(n, k)
        if self.treatment in ("deepsets", "transformer"):
            slots = torch.arange(k, device=h.device, dtype=h.dtype)[None, :]
            position = slots / (palette.to(h.dtype) - 1).clamp_min(1)[:, None]
            inverse_k = (1.0 / palette.to(h.dtype))[:, None].expand(n, k)
            if self.treatment == "deepsets":
                slot_state = torch.cat((h, torch.stack((alpha, position, inverse_k), -1)), -1)
                phi = self.palette_phi(slot_state) * in_palette[:, :, None]
                context = phi.sum(1) / palette[:, None]
                features = torch.cat((h, torch.stack((alpha, position, inverse_k, r), -1),
                                      context[:, None, :].expand(n, k, -1)), -1)
            else:
                local = torch.cat((h, torch.stack((alpha, position, inverse_k, r), -1)), -1)
                features = self.palette_attention(self.palette_project(local),
                                                   src_key_padding_mask=~in_palette)
            return self.scorer(features).squeeze(-1).masked_fill(~in_palette, NEG)
        khat = alpha.sum(1)
        prefix = torch.cumsum(alpha, 1) - alpha
        denominator = khat.clamp_min(.5)[:, None]
        # This conditional is the complete normalization intervention.
        lo, hi = ((prefix, prefix + alpha) if self.treatment == "raw_endpoints"
                  else (prefix / denominator, (prefix + alpha) / denominator))
        extra = torch.stack((alpha, lo, hi, r), -1)
        return self.scorer(torch.cat((h, extra), -1)).squeeze(-1).masked_fill(~in_palette, NEG)

    def stage1(self, colors, rp, rc, edge_index, palette, edge_chunk=None):
        _, in_palette, own, uncolored = self._slots(colors, palette)
        n, k = own.shape
        u = uncolored.float()
        f1 = torch.stack((u[:, None].expand(n, k), own), -1)
        m = self.message1(f1)
        d = m.shape[-1]
        received = neighbor_reduce(m.reshape(n, k * d), edge_index, self.reduction, edge_chunk).reshape(n, k, d)
        h = self.state1(torch.cat((u[:, None, None].expand(n, k, 1), received), -1))
        alpha = torch.sigmoid(self.availability(h).squeeze(-1)) * in_palette
        khat = alpha.sum(1)
        prefix = torch.cumsum(alpha, 1) - alpha
        lo, hi = prefix / khat.clamp_min(.5)[:, None], (prefix + alpha) / khat.clamp_min(.5)[:, None]
        if self.treatment in ("deepsets", "transformer"):
            # Diagnostic lo/hi above have no path to these scores.
            scores = self.palette_scores(h, alpha, rc, palette, in_palette)
        else:
            r = rc[:, None].expand(n, k)
            endpoints = (prefix, prefix + alpha) if self.treatment == "raw_endpoints" else (lo, hi)
            extra = (alpha, *endpoints, r)
            scores = self.scorer(torch.cat((h, torch.stack(extra, dim=-1)), dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~in_palette, NEG)
        p = self.participation(torch.stack((rp, u), -1)).squeeze(-1)
        return p, scores, alpha, khat, lo, hi

    def commit_logits(self, colors, edge_index, palette, participate, candidate, edge_chunk=None):
        if self.reduction == "max":
            return super().commit_logits(colors, edge_index, palette, participate, candidate, edge_chunk)
        slots, _, own, uncolored = self._slots(colors, palette)
        n, k = own.shape
        candidate = candidate.to(colors.device, torch.long)
        participate = participate.to(colors.device).bool() & uncolored
        candidate = torch.where(uncolored, candidate, torch.full_like(candidate, -1))
        if bool(((candidate < -1) | (candidate >= palette.to(colors.device))).any()):
            raise ValueError("Candidates must be -1 or inside the declared palette")
        offer = (candidate[:, None] == slots).float()
        u, p = uncolored.float(), participate.float()
        f = torch.stack((u[:, None].expand(n, k), own, p[:, None].expand(n, k), offer), -1)
        m = self.message2(f)
        d = m.shape[-1]
        received = neighbor_reduce(m.reshape(n, k * d), edge_index, self.reduction, edge_chunk).reshape(n, k, d)
        z = self.state2(torch.cat((f, received), -1))
        gathered = (offer[:, :, None] * z).sum(1)
        return self.commit_head(torch.cat((gathered, p[:, None], u[:, None], offer.sum(1, keepdim=True)), -1)).squeeze(-1)


class LocalPaletteTransformer(MechanismUCGNN):
    """Two neighbor communications and self-attention over each local palette."""
    def __init__(self, hidden_dim=64, message_dim=32):
        super().__init__("transformer", hidden_dim, message_dim)


def make_model(treatment):
    return LocalPaletteTransformer() if treatment == "transformer" else MechanismUCGNN(treatment)
