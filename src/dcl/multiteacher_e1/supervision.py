"""Concordant native labels; never imported by model or autonomous inference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dcl.multiteacher.programs import DETECT, KW, LINIAL, MAX_FIELD


@dataclass
class Targets:
    participate: np.ndarray
    candidate: np.ndarray
    free: np.ndarray
    winning: np.ndarray | None
    full_conflicts: bool


def labels_from_inputs(arrays, labels, teacher):
    """Root/neighbor labels from the same source state and native random tape.

    The complete neighbors of each first-hop vertex are present, so JP labels
    and availability have no truncated-neighborhood approximation.
    """
    x, y, rel, masks = arrays
    phase = int(labels[0, 3])
    if not np.all(labels[:, 3] == phase):
        raise ValueError("Group native phases before constructing labels")
    permission = x[..., 7] > .5
    free = ~((y[..., 1:5] > .5) & masks[..., None]).any(2)
    if phase == KW:
        free = ~((rel[..., -4:] > .5) & masks[..., None]).any(2)
    candidate = free.argmax(-1)
    participate = permission.copy()
    winning = None
    if teacher in ("LUBY", "JOHANSSON"):
        rank = np.floor(x[..., 33] * free.sum(-1)).astype(int)
        candidate = (free & (free.cumsum(-1) - 1 == rank[..., None])).argmax(-1)
        participate &= x[..., 32] < (.5 if teacher == "LUBY" else 1.)
    elif teacher == "JP":
        pending = y[..., 0] > .5
        larger = y[..., 5] > x[..., None, 5]
        participate &= ~(masks & pending & larger).any(2)
    elif teacher == "COLOR_TRIALS":
        prefs = np.where(free & permission[..., None], x[..., 34:38], -1.)
        other = np.where(masks[:, 0, :, None], prefs[:, 1:], -1.).max(1)
        winning = free[:, 0] & (prefs[:, 0] > other)
        candidate[:, 0] = np.where(winning.any(1), winning.argmax(1), candidate[:, 0])
    elif phase == KW:
        participate &= x[..., 18] >= 0
    if phase == LINIAL:
        candidate[:, 0] = labels[:, 1] // MAX_FIELD
    result = Targets(participate, candidate, free[:, 0], winning,
                     teacher in ("LUBY", "JOHANSSON", "JP"))
    if phase not in (DETECT, LINIAL):
        expected = np.where(commit_targets(result, participate, candidate), candidate[:, 0] + 1, 0)
        if not np.array_equal(expected, labels[:, 0]):
            raise RuntimeError("E1 decomposition does not reproduce native teacher decisions")
    return result


def commit_targets(targets, participate, candidate):
    b = len(candidate)
    root_candidate = candidate[:, 0]
    available = targets.free[np.arange(b), root_candidate]
    if targets.winning is not None:
        available &= targets.winning[np.arange(b), root_candidate]
    safe = participate[:, 0] & available
    if targets.full_conflicts:
        safe &= ~(participate[:, 1:] & (candidate[:, 1:] == root_candidate[:, None])).any(1)
    return safe


def counterfactual(targets, arrays, teacher, rng):
    """Star-local counterfactuals with fixed (0.17) and edge (0.04) injections.

    Each sampled root is a separate decision context. Labels below are always
    recomputed for the actual proposals. This uses the E1 loss, but does not
    claim the full-graph injection distribution of canonical E1 training.
    Native ColorTrials/KW also receive uniformly changed candidate queries.
    """
    x, y, rel, masks = arrays
    p, c = targets.participate.copy(), targets.candidate.copy()
    b, centers = p.shape
    if teacher in ("COLOR_TRIALS", "BE_BASE_D3"):
        change = rng.random(b) < .5
        c[change, 0] = rng.integers(4, size=int(change.sum()))
        p[change, 0] = True
        return p, c
    # Inject fixed-neighbor colors into root/neighbor proposals. The resulting
    # four proposals suffice for the root's COMMIT decision.
    for edge in range(masks.shape[2]):
        fixed = y[:, :, edge, 1:5].sum(-1) > .5
        hit = masks[:, :, edge] & fixed & (rng.random((b, centers)) < .17) & (x[..., 7] > .5)
        colors = y[:, :, edge, 1:5].argmax(-1)
        p[hit], c[hit] = True, colors[hit]
    for edge in range(centers - 1):
        hit = masks[:, 0, edge] & (x[:, 0, 7] > .5) & (x[:, edge + 1, 7] > .5)
        hit &= rng.random(b) < .04
        chosen = rng.integers(4, size=int(hit.sum()))
        p[hit, 0] = p[hit, edge + 1] = True
        c[hit, 0] = c[hit, edge + 1] = chosen
    return p, c


def tensor_arrays(arrays):
    return tuple(torch.from_numpy(a) for a in arrays)


def phase_groups(arrays, labels):
    for phase in np.unique(labels[:, 3]):
        use = labels[:, 3] == phase
        yield int(phase), tuple(a[use] for a in arrays), labels[use]


def loss_one(model, arrays, labels, warmup, rng):
    phase = int(labels[0, 3])
    tensors = tensor_arrays(arrays)
    out = model(*tensors, phase=phase)
    device = out.commit.device
    if phase == DETECT:
        loss = F.binary_cross_entropy_with_logits(out.commit, torch.from_numpy(labels[:, 2].astype(np.float32)).to(device))
        return loss, out
    targets = labels_from_inputs(arrays, labels, model.teacher)
    c_loss = F.cross_entropy(out.scores, torch.from_numpy(targets.candidate[:, 0]).to(device))
    if phase == LINIAL:
        return c_loss, out
    p_loss = F.binary_cross_entropy_with_logits(out.state.p_logits[:, 0],
                                                torch.from_numpy(targets.participate[:, 0].astype(np.float32)).to(device))
    if warmup:
        p, c = torch.from_numpy(targets.participate).to(device), torch.from_numpy(targets.candidate).to(device)
        logits = model.commit_for(out.state, p, c)
        truth = commit_targets(targets, targets.participate, targets.candidate)
        own_loss = cf_loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(truth.astype(np.float32)).to(device))
    else:
        own_p, own_c = out.p.detach().cpu().numpy(), out.candidates.detach().cpu().numpy()
        truth = commit_targets(targets, own_p, own_c)
        own_loss = F.binary_cross_entropy_with_logits(out.commit, torch.from_numpy(truth.astype(np.float32)).to(device))
        cf_p, cf_c = counterfactual(targets, arrays, model.teacher, rng)
        truth = commit_targets(targets, cf_p, cf_c)
        logits = model.commit_for(out.state, torch.from_numpy(cf_p).to(device), torch.from_numpy(cf_c).to(device))
        cf_loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(truth.astype(np.float32)).to(device))
    return p_loss + c_loss + .5 * own_loss + .5 * cf_loss, out
