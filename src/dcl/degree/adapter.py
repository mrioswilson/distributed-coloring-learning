"""Sparse evaluation of frozen RootedPalettePolicy weights at variable K/degree.

Only raw color predicates, private inputs, permissions, phases and the same
signed ID/slot relations enter the MLPs. No exact availability, conflict,
eligibility or teacher action is computed here. Category support is an exact
compression of MAX over nonnegative, finite-alphabet first messages.
"""
from dataclasses import dataclass
import numpy as np
import torch

from dcl.multiteacher.programs import Choices, DETECT
from dcl.multiteacher_e1.model import binary_table


@dataclass
class Decision:
    choices: Choices
    roots: np.ndarray
    participation: np.ndarray
    candidates: np.ndarray
    commit: np.ndarray
    p_logits: np.ndarray
    commit_logits: np.ndarray
    availability: np.ndarray
    scores: np.ndarray


class FrozenAdapter:
    def __init__(self, model):
        self.model = model.eval()
        self.m1 = binary_table(model.core.message1, 2)
        self.m2 = binary_table(model.core.message2, 4)

    @torch.inference_mode()
    def __call__(self, program, state, overrides=None):
        model, n, k = self.model, program.n, program.k
        roots = np.flatnonzero(state.permission)
        if not len(roots):
            raise ValueError("No writable roots")
        adjacent = program.neighbors[roots].ravel()
        centers = np.unique(np.r_[roots, adjacent[adjacent < n]])
        m, nr = len(centers), len(roots)
        reverse = np.full(n, -1, np.int64)
        reverse[centers] = np.arange(m)
        rpos = torch.from_numpy(reverse[roots])
        slots = np.arange(k)
        u_all = state.colors < 0
        if program.teacher == "SPECULATIVE_GM":
            u_all = u_all | state.permission
        u = torch.from_numpy(u_all[centers].astype(np.float32))
        own_np = state.colors[centers, None] == slots
        own = torch.from_numpy(own_np.astype(np.float32))
        valid = torch.from_numpy(slots[None, :] < program.palette[centers, None])
        phase = torch.zeros(6)
        phase[state.phase] = 1
        local = torch.cat((phase[None].expand(m, 6),
                           torch.from_numpy(state.permission[centers, None].astype(np.float32)),
                           torch.zeros((m, 1))), -1)
        src, dst = program.edges
        use = reverse[dst] >= 0
        src, dst = src[use], dst[use]
        receiver = reverse[dst]
        id_dependent = program.teacher in ("JP", "SPECULATIVE_GM")
        difference = np.sign(state.ids[src] - state.ids[dst]) if id_dependent else np.zeros(len(src), np.int64)
        same_slot = state.slots[src] == state.slots[dst] if program.teacher == "SPECULATIVE_GM" else np.zeros(len(src), bool)
        context_code = (difference + 1) * 2 + same_slot.astype(int)
        received = torch.zeros((m, k, self.m1.shape[-1]))
        present_codes = np.zeros((m, k), np.int64)
        # category=(pending,owns receiver color); multiplicities disappear under
        # the original MAX too. This retains every observed category.
        for ctx in np.unique(context_code):
            ctx_edges = context_code == ctx
            if model.native:
                context = torch.cat((phase, torch.tensor([float(ctx // 2 - 1), float(ctx % 2)])))
                residual = model.context_message1(context)
            else:
                residual = 0
            for pending in (0, 1):
                pick = ctx_edges & (u_all[src] == pending)
                counts = np.bincount(receiver[pick], minlength=m)
                same = np.zeros((m, k), np.int64)
                colored = pick & (state.colors[src] >= 0) & (state.colors[src] < k)
                np.add.at(same, (receiver[colored], state.colors[src[colored]]), 1)
                for owns, existence in ((0, counts[:, None] > same), (1, same > 0)):
                    code = pending * 2 + owns
                    if model.native:
                        message = self.m1[code] + residual
                        received = torch.maximum(received, torch.from_numpy(existence[..., None]) * message)
                    else:
                        present_codes |= existence.astype(np.int64) * (1 << code)
        if model.native:
            h = model.core.state1(torch.cat((u[:, None, None].expand(m, k, 1), received), -1))
            h = torch.relu(h + model.context_state1(local)[:, None, :])
        else:
            members = ((torch.arange(16)[:, None] // (2 ** torch.arange(4))) % 2).float()
            table = (self.m1[None] * members[..., None]).amax(1)
            states = torch.arange(32)
            h_table = model.core.state1(torch.cat(((states // 16).float()[:, None], table[states % 16]), -1))
            h = h_table[u.long()[:, None] * 16 + torch.from_numpy(present_codes)]
        rc = torch.from_numpy(state.draws[centers, 1])
        scores, alpha = model._scores(h, valid, rc)
        p_logits = model.core.participation(torch.stack((torch.from_numpy(state.draws[centers, 0]), u), -1)).squeeze(-1)
        if model.native:
            # Padding is excluded, just as in K4. Global palettes reduce to
            # h.mean(K) exactly; local lists use their own valid entries.
            if bool(valid.all()):
                mean_h = h.mean(1)
            else:
                mean_h = (h * valid[..., None]).sum(1) / valid.sum(1)[:, None]
            p_logits += model.context_participation(torch.cat((mean_h, local), -1)).squeeze(-1)
        p = (p_logits > 0) & torch.from_numpy(state.permission[centers])
        candidate = scores.argmax(-1)
        if state.phase == DETECT:
            candidate = own.argmax(-1)
            p = torch.from_numpy(state.permission[centers])
        if overrides is not None:
            if "participation" in overrides:
                p = torch.from_numpy(np.asarray(overrides["participation"], bool)[centers])
            if "candidates" in overrides:
                candidate = torch.from_numpy(np.asarray(overrides["candidates"], np.int64)[centers])
        # All neighbors of every queried root belong to centers.
        roots_map = np.full(n, -1, np.int64)
        roots_map[roots] = np.arange(nr)
        edge_mask = roots_map[dst] >= 0
        sidx = torch.from_numpy(reverse[src[edge_mask]])
        ridx = torch.from_numpy(roots_map[dst[edge_mask]])
        ediff = torch.from_numpy(difference[edge_mask].astype(np.float32))
        eslot = torch.from_numpy(same_slot[edge_mask].astype(np.float32))
        ct = program.teacher == "COLOR_TRIALS"
        if ct:
            own_root = own[rpos]
            offers_root = torch.zeros_like(own_root)
            froot = torch.stack((u[rpos, None].expand(nr, k), own_root,
                                p[rpos, None].float().expand(nr, k), offers_root), -1)
            received2 = torch.zeros((nr, k, self.m2.shape[-1]))
            pref_bound = 3 if program.palette_regime == "native4" else max(1, program.delta)
            prefs = torch.from_numpy(state.preferences[centers] / pref_bound)
            for start in range(0, len(sidx), max(1, 32768 // k)):
                si, ri = sidx[start:start+max(1, 32768//k)], ridx[start:start+max(1, 32768//k)]
                fneighbor = torch.stack((u[si, None].expand(-1, k), own[si],
                                        p[si, None].float().expand(-1, k), torch.zeros((len(si), k))), -1)
                codes = (fneighbor.long() * torch.tensor([8, 4, 2, 1])).sum(-1)
                colors = torch.stack((prefs[si], prefs[rpos[ri]], alpha[si], alpha[rpos[ri]]), -1)
                contexts = torch.cat((phase[None, None].expand(len(si), k, 6),
                                      torch.zeros((len(si), k, 2)), colors), -1)
                messages = self.m2[codes] + model.context_message2(contexts)
                received2.scatter_reduce_(0, ri[:, None, None].expand_as(messages), messages,
                                          reduce="amax", include_self=True)
            z = model.core.state2(torch.cat((froot, received2), -1))
            z = torch.relu(z + model.context_state2(local[rpos])[:, None])
            final_scores, _ = model._scores(z, valid[rpos], rc[rpos])
            query = final_scores.argmax(-1)
            if overrides is not None and "query" in overrides:
                # Declared evaluation-only intervention on COMMIT's query;
                # never used in an autonomous rollout.
                query = torch.from_numpy(np.asarray(overrides["query"], np.int64)[roots])
                if np.any((query.numpy() < 0) | (query.numpy() >= program.palette[roots])):
                    raise ValueError("Counterfactual query outside receiver palette")
            gathered = z[torch.arange(nr), query]
        else:
            query = candidate[rpos]
            # Querying one receiver color is an exact restriction of stage2,
            # not an availability/conflict computation.
            froot = torch.stack((u[rpos], own[rpos, query], p[rpos].float(), (u[rpos] > .5).float()), -1)
            received2 = torch.zeros((nr, self.m2.shape[-1]))
            fneighbor = torch.stack((u[sidx], own[sidx, query[ridx]], p[sidx].float(),
                                    ((candidate[sidx] == query[ridx]) & (u[sidx] > .5)).float()), -1)
            codes = (fneighbor.long() * torch.tensor([8, 4, 2, 1])).sum(-1)
            messages = self.m2[codes]
            if model.native:
                contexts = torch.cat((phase[None].expand(len(sidx), 6), ediff[:, None],
                                      eslot[:, None], torch.zeros((len(sidx), 4))), -1)
                messages += model.context_message2(contexts)
            received2.scatter_reduce_(0, ridx[:, None].expand_as(messages), messages, reduce="amax", include_self=True)
            gathered = model.core.state2(torch.cat((froot, received2), -1))
            if model.native:
                gathered = torch.relu(gathered + model.context_state2(local[rpos]))
            final_scores = scores[rpos]
        commit_logit = model.core.commit_head(torch.cat((gathered, p[rpos, None].float(),
                                                       u[rpos, None], torch.ones((nr, 1))), -1)).squeeze(-1)
        commit = commit_logit > 0
        color, mark = np.zeros(n, np.int64), np.zeros(n, bool)
        if state.phase == DETECT:
            mark[roots] = commit.numpy()
        else:
            color[roots] = torch.where(p[rpos] & commit, query + 1, 0).numpy()
        return Decision(Choices(color, np.zeros(n, np.int64), mark), roots, p[rpos].numpy(),
                        query.numpy(), commit.numpy(), p_logits[rpos].numpy(), commit_logit.numpy(),
                        alpha[rpos].numpy(), final_scores.numpy())
