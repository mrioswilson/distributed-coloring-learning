"""Receiver-color restriction of frozen modules; no exact-availability shield.

Only the RECEIVER's valid colors are evaluated. Sender state predicates remain
defined at any receiver color, even outside the sender's own list. MAX's finite
first-message alphabet is evaluated exactly; SUM/MEAN preserve incidence order.
"""
from __future__ import annotations

import numpy as np
import torch

from dcl.universal.model import NEG, UOutput


def converted_uniform(x):
    return np.minimum(np.asarray(x).astype(np.float32), np.nextafter(np.float32(1), np.float32(0)))


def first_state(model, colors, edges, receivers, k, edge_chunk=512):
    """First communication at exactly k valid receiver colors."""
    device, dtype = colors.device, next(model.parameters()).dtype
    slots = torch.arange(k, device=device)
    src, dst = edges
    mapping = torch.full_like(colors, -1)
    mapping[receivers] = torch.arange(len(receivers), device=device)
    use = mapping[dst] >= 0
    src, dst = src[use], mapping[dst[use]]
    n = len(receivers)
    table = model.message1(torch.tensor([[0, 0], [0, 1], [1, 0]], device=device, dtype=dtype))
    d = table.shape[-1]
    reduction = getattr(model, 'reduction', 'max')
    if reduction == 'max':
        # Exact reduction of the three legal [uncolored,own] message types.
        pending = torch.bincount(dst[colors[src] < 0], minlength=n)
        fixed = torch.bincount(dst[colors[src] >= 0], minlength=n)
        same = torch.zeros((n, k), device=device, dtype=torch.long)
        fixed_inside = (colors[src] >= 0) & (colors[src] < k)
        linear = dst[fixed_inside] * k + colors[src[fixed_inside]]
        same.view(-1).index_add_(0, linear, torch.ones_like(linear))
        out = torch.maximum(
            (fixed[:, None, None] > same[:, :, None]).to(dtype) * table[0],
            (same[:, :, None] > 0).to(dtype) * table[1])
        out = torch.maximum(out, (pending[:, None, None] > 0).to(dtype) * table[2])
    else:
        out = torch.zeros((n, k, d), device=device, dtype=dtype)
        for start in range(0, len(src), edge_chunk):
            s, t = src[start:start+edge_chunk], dst[start:start+edge_chunk]
            typ = 2 * (colors[s] < 0).long()[:, None] + (colors[s, None] == slots).long()
            out.index_add_(0, t, table[typ])
        if reduction == 'mean':
            out = out / torch.bincount(dst, minlength=n).clamp_min(1)[:, None, None]
    u = (colors[receivers] < 0).to(dtype)
    h = model.state1(torch.cat((u[:, None, None].expand(n, k, 1), out), -1))
    alpha = torch.sigmoid(model.availability(h).squeeze(-1))
    return h, alpha


def score(model, h, alpha, rc):
    n, k = alpha.shape
    if not hasattr(model,'palette_scores'):
        total=alpha.sum(1).clamp_min(.5)[:,None]
        prefix=alpha.cumsum(1)-alpha
        extra=torch.stack((alpha,prefix/total,(prefix+alpha)/total,rc[:,None].expand(n,k)),-1)
        return model.scorer(torch.cat((h,extra),-1)).squeeze(-1)
    palette = torch.full((n,), k, device=h.device, dtype=torch.long)
    mask = torch.ones((n, k), device=h.device, dtype=torch.bool)
    return model.palette_scores(h, alpha, rc, palette, mask)


def score_chunk(model, k):
    return max(1, min(128, 4_000_000 // (2*k*k))) if getattr(model,'treatment','') == 'transformer' else 256


class LocalPaletteAdapter:
    def __init__(self, model):
        self.model = model
        self._cache = None

    @torch.inference_mode()
    def __call__(self, colors, rp, rc, edges, palette, output_width=None):
        model = self.model
        if len(colors) == 0 or (palette < 1).any() or ((colors < -1) | (colors >= palette)).any():
            raise ValueError('Invalid colors or local palettes')
        width = int(palette.max()) if output_width is None else int(output_width)
        if width < int(palette.max()):
            raise ValueError('Storage width smaller than a local palette')
        dtype, device = next(model.parameters()).dtype, colors.device
        n = len(colors)
        scores = torch.full((n, width), NEG, dtype=dtype, device=device)
        alpha = torch.zeros_like(scores)
        lo, hi = torch.zeros_like(scores), torch.zeros_like(scores)
        khat = torch.zeros(n, dtype=dtype, device=device)
        same = self._cache is not None and all(torch.equal(x,y) for x,y in zip(self._cache[:3],(colors,edges,palette)))
        if not same:
            groups=[]
            for kval in torch.unique(palette).tolist():
                indices=torch.nonzero(palette==kval).flatten()
                h,a=first_state(model,colors,edges,indices,int(kval))
                groups.append((int(kval),indices,h,a))
            self._cache=(colors.clone(),edges.clone(),palette.clone(),groups)
        # Cached first communication is independent of BOTH random inputs.
        for k,indices,h,a in self._cache[3]:
            total = a.sum(1)
            prefix = a.cumsum(1) - a
            den = total.clamp_min(.5)[:, None]
            khat[indices] = total
            alpha[indices, :k], lo[indices, :k], hi[indices, :k] = a, prefix/den, (prefix+a)/den
            step = score_chunk(model, k)
            for start in range(0, len(indices), step):
                ids = indices[start:start+step]
                scores[ids, :k] = score(model, h[start:start+step], a[start:start+step], rc[ids])
        u = (colors < 0).to(dtype)
        p_logit = model.participation(torch.stack((rp, u), -1)).squeeze(-1)
        participate = (p_logit > 0) & (colors < 0)
        candidate = torch.where(colors < 0, scores.argmax(1), -1)
        commit = self.commit_logits(colors, edges, palette, participate, candidate)
        return UOutput(p_logit, scores, commit, participate, candidate, alpha, khat, lo, hi)

    @torch.inference_mode()
    def commit_logits(self, colors, edges, palette, participate, candidate):
        model = self.model
        dtype = next(model.parameters()).dtype
        if ((candidate < -1) | (candidate >= palette)).any():
            raise ValueError('Candidate outside own palette')
        u = colors < 0
        p = participate.bool() & u
        candidate = torch.where(u, candidate, -1)
        query = candidate.clamp_min(0)
        src, dst = edges
        f = torch.stack((u[src], colors[src] == query[dst], p[src], candidate[src] == query[dst]), -1).to(dtype)
        m = model.message2(f)
        received = torch.zeros((len(colors), m.shape[1]), device=colors.device, dtype=dtype)
        reduction = getattr(model, 'reduction', 'max')
        if reduction == 'max':
            received.scatter_reduce_(0, dst[:, None].expand_as(m), m, reduce='amax', include_self=True)
        else:
            received.index_add_(0, dst, m)
            if reduction == 'mean':
                received /= torch.bincount(dst, minlength=len(colors)).clamp_min(1)[:, None]
        offer = candidate >= 0
        own = colors == query
        own_features = torch.stack((u, own, p, offer), -1).to(dtype)
        z = model.state2(torch.cat((own_features, received), -1)) * offer[:, None]
        return model.commit_head(torch.cat((z, p[:, None], u[:, None], offer[:, None]), -1).to(dtype)).squeeze(-1)


@torch.inference_mode()
def choices(model, state, uniforms):
    """Cache only deterministic first-communication state; fresh rc per sample."""
    h, a = state
    k = len(a)
    out = []
    for start in range(0, len(uniforms), score_chunk(model, k)):
        r = torch.as_tensor(uniforms[start:start+score_chunk(model, k)], device=h.device, dtype=h.dtype)
        b = len(r)
        out.append(score(model, h[None].expand(b, k, -1), a[None].expand(b, k), r).argmax(1).cpu().numpy())
    return np.concatenate(out)
