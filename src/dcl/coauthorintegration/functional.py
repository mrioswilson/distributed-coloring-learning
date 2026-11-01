"""Component interventions on frozen BE models, with the native evaluator.

Oracle calls below are deliberately diagnostic. They must never be described
as autonomous learned inference or used to rewrite an original result.
"""
from collections import Counter
from dataclasses import replace

import numpy as np
import torch

from dcl.multicf.oracles import confirmation
from dcl.multiteacher.programs import KW


MODES = ("baseline", "commit", "participation", "candidate", "kw_all")


def kw_reference_messages(arrays):
    """Exact P and first-free option for each represented center, not decoding."""
    x, _, rel, masks = arrays
    p = (x[..., 7] > .5) & (x[..., 18] >= 0)
    free = ~((rel[..., -4:] > .5) & masks[..., None]).any(2)
    c = free.long().argmax(-1)
    return p, c


class BEComponentPolicy(torch.nn.Module):
    def __init__(self, learned, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.learned, self.mode = learned, mode
        self.counts = Counter()
        self.strata = {}

    @torch.inference_mode()
    def forward(self, *arrays, phase=0):
        out = self.learned(*arrays, phase=phase)
        if phase != KW:
            return out
        raw = tuple(a.detach().numpy() for a in arrays)
        p0, c0 = out.p.detach().numpy(), out.candidates.detach().numpy()
        truth = confirmation(raw, "BE_BASE_D3", p0, c0) > .5
        pred = out.commit.detach().numpy() > 0
        eligible = (raw[0][:, 0, 7] > .5) & (raw[0][:, 0, 18] >= 0)
        active = p0[:, 0].astype(bool)
        occupied_slots = ((raw[2][:, 0, :, -4:] > .5) & raw[3][:, 0, :, None]).any(1)
        occupied = occupied_slots[np.arange(len(p0)), c0[:, 0]]
        boundary = raw[0][:, 0, 18] == 0
        neighbor_active = (p0[:, 1:] & raw[3][:, 0]).any(1)
        for name, mask in (
            ("eligible_active_free", eligible & active & ~occupied),
            ("eligible_active_occupied", eligible & active & occupied),
            ("eligible_silent", eligible & ~active),
            ("ineligible_active", ~eligible & active),
            ("ineligible_silent", ~eligible & ~active),
        ):
            d = self.strata.setdefault(name, Counter())
            d.update(n=int(mask.sum()), tp=int((mask & pred & truth).sum()),
                     tn=int((mask & ~pred & ~truth).sum()),
                     fp=int((mask & pred & ~truth).sum()), fn=int((mask & ~pred & truth).sum()))
        fn = ~pred & truth
        self.counts.update(
            kw_root_queries=len(p0),
            learned_masked_false_positives=int((pred & ~truth & ~active).sum()),
            learned_unsafe_accepts=int((pred & active & ~truth).sum()),
            learned_missed_valid_writes=int(fn.sum()),
            missed_valid_writes_at_boundary=int((fn & boundary).sum()),
            missed_valid_writes_with_active_neighbor=int((fn & neighbor_active).sum()),
            upstream_eligible_silence=int((eligible & ~active).sum()),
            upstream_active_occupied_proposals=int((eligible & active & occupied).sum()),
        )
        if self.mode == "baseline":
            return out
        p, c = out.p, out.candidates
        if self.mode in ("participation", "candidate", "kw_all"):
            exact_p, exact_c = kw_reference_messages(arrays)
            if self.mode in ("participation", "kw_all"):
                p = exact_p
            if self.mode in ("candidate", "kw_all"):
                c = exact_c
        if self.mode in ("commit", "kw_all"):
            y = confirmation(raw, "BE_BASE_D3", p.numpy(), c.numpy())
            logits = torch.from_numpy(np.where(y > .5, 1., -1.).astype(np.float32))
        else:
            logits = self.learned.commit_for(out.state, p, c)
        return replace(out, p=p, candidates=c, commit=logits)

    def diagnostics(self):
        return dict(scope="Learned decisions on visited states before this mode's component override; "
                          "different modes generally visit different states",
                    counts=dict(self.counts), strata={k: dict(v) for k, v in self.strata.items()})
