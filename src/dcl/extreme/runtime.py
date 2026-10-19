"""Inference-only color streaming of the unchanged two-communication network.

Every message/MLP is evaluated with the original weights. This trades a
second evaluation of stage-1 messages for bounded [N, color_chunk, D]
activations. No availability predicate or teacher operation is introduced.
"""
from __future__ import annotations

import torch

from dcl.universal.model import NEG, UOutput


def sparse_max(z, edges, chunk):
    out = torch.zeros_like(z)
    for start in range(0, edges.shape[1], chunk):
        s, d = edges[:, start:start+chunk]
        out.scatter_reduce_(0, d[:,None].expand(-1,z.shape[1]), z[s], reduce="amax", include_self=True)
    return out


class ChunkedPolicy:
    def __init__(self, model, color_chunk=8, edge_budget_bytes=32*2**20):
        self.model = model
        self.color_chunk = color_chunk
        self.edge_budget_bytes = edge_budget_bytes
        self.message1 = model.message1  # rollout auto-chunk interface

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def eval(self):
        self.model.eval()
        return self

    def _state1(self, colors, edges, start, end, u):
        slots = torch.arange(start,end,device=colors.device)
        own = (colors[:,None] == slots).float()
        f = torch.stack((u[:,None].expand_as(own),own),-1)
        m = self.model.message1(f)
        n,b,d = m.shape
        ec = max(1,self.edge_budget_bytes//(b*d*m.element_size()))
        received = sparse_max(m.reshape(n,b*d),edges,ec).reshape(n,b,d)
        return self.model.state1(torch.cat((u[:,None,None].expand(n,b,1),received),-1))

    @torch.inference_mode()
    def stage1(self, colors, rp, rc, edges, palette, edge_chunk=None):
        # _slots preserves the original range checks and public palette mask.
        slots,in_palette,_,uncolored = self.model._slots(colors,palette)
        n,k = in_palette.shape
        u = uncolored.float()
        alpha = torch.empty((n,k),device=colors.device)
        for start in range(0,k,self.color_chunk):
            end = min(k,start+self.color_chunk)
            h = self._state1(colors,edges,start,end,u)
            alpha[:,start:end] = torch.sigmoid(self.model.availability(h).squeeze(-1)) * in_palette[:,start:end]
        k_hat = alpha.sum(1)
        prefix = torch.cumsum(alpha,1)-alpha
        denom = k_hat.clamp_min(.5)[:,None]
        lo,hi = prefix/denom,(prefix+alpha)/denom
        scores = torch.empty_like(alpha)
        for start in range(0,k,self.color_chunk):
            end = min(k,start+self.color_chunk)
            h = self._state1(colors,edges,start,end,u)
            a = alpha[:,start:end]
            r = rc[:,None].expand(n,end-start)
            if self.model.variant == "NORMALIZED":
                extra = (a,lo[:,start:end],hi[:,start:end],r)
            else:
                extra = (a,prefix[:,start:end],(prefix+alpha)[:,start:end],k_hat[:,None].expand_as(a),r)
            scores[:,start:end] = self.model.scorer(torch.cat((h,torch.stack(extra,-1)),-1)).squeeze(-1)
        scores.masked_fill_(~in_palette,NEG)
        plogit = self.model.participation(torch.stack((rp,u),-1)).squeeze(-1)
        return plogit,scores,alpha,k_hat,lo,hi

    @torch.inference_mode()
    def commit_logits(self, colors, edges, palette, participate, candidate, edge_chunk=None):
        _,_,_,uncolored = self.model._slots(colors,palette)
        candidate = candidate.to(colors.device,torch.long)
        candidate = torch.where(uncolored,candidate,torch.full_like(candidate,-1))
        if bool(((candidate < -1)|(candidate >= palette)).any()):
            raise ValueError("Candidates must be inside the public palette")
        u,p = uncolored.float(),(participate.bool() & uncolored).float()
        n,k = len(colors),int(palette.max())
        d = self.model.message2[0].out_features
        gathered = torch.zeros((n,d),device=colors.device)
        offered = torch.zeros((n,1),device=colors.device)
        for start in range(0,k,self.color_chunk):
            end = min(k,start+self.color_chunk)
            slots = torch.arange(start,end,device=colors.device)
            own = (colors[:,None] == slots).float()
            offer = (candidate[:,None] == slots).float()
            f = torch.stack((u[:,None].expand_as(own),own,p[:,None].expand_as(own),offer),-1)
            m = self.model.message2(f)
            b = end-start
            ec = max(1,self.edge_budget_bytes//(b*d*m.element_size()))
            received = sparse_max(m.reshape(n,b*d),edges,ec).reshape(n,b,d)
            z = self.model.state2(torch.cat((f,received),-1))
            gathered += (offer[:,:,None]*z).sum(1)
            offered += offer.sum(1,keepdim=True)
        return self.model.commit_head(torch.cat((gathered,p[:,None],u[:,None],offered),-1)).squeeze(-1)

    @torch.inference_mode()
    def __call__(self, colors, rp, rc, edges, palette, edge_chunk=None):
        plogit,scores,alpha,khat,lo,hi = self.stage1(colors,rp,rc,edges,palette)
        uncolored = colors < 0
        p = (plogit > 0) & uncolored
        c = torch.where(uncolored,scores.argmax(1),torch.full_like(colors,-1))
        commit = self.commit_logits(colors,edges,palette,p,c)
        return UOutput(plogit,scores,commit,p,c,alpha,khat,lo,hi)


def estimate_cost(n, degree, palette, color_chunk=8, message_dim=32):
    """Conservative live tensor allowance; a bound estimate, not measured VRAM.

    NetworkX/diagnostics are CPU-side. E counts directed edges; dense NxN
    adjacency is never allocated. Edge messages are bounded separately.
    """
    directed_edges = n*degree
    nk = n*palette
    block = n*min(color_chunk,palette)
    estimated = 384*2**20 + 4*(12*nk + 14*block*message_dim) + 16*directed_edges
    return {"estimated_gpu_bytes": estimated, "node_color_representations": nk,
            "max_directed_edges": directed_edges, "message_dim": message_dim,
            "message_floats_per_directed_edge_per_stage": palette*message_dim,
            "logical_message_bytes_per_round": 2*directed_edges*palette*message_dim*4,
            "streamed_aggregation_message_bytes_per_round": 3*directed_edges*palette*message_dim*4,
            "estimated_gpu_bytes_per_node": estimated/n,
            "original_single_NKD_tensor_bytes": nk*message_dim*4,
            "edge_chunk_budget_bytes": 32*2**20, "color_chunk": color_chunk}
