"""Counterfactual proposal supervision for M4B."""

import numpy as np
import torch
import torch.nn.functional as F

from dcl.staged_models import neighbor_max


def commit_logits_for_proposals(
    model, x, edge_index, participate, candidate_class
):
    """
    Evaluate M4B's original COMMIT pathway using
    externally supplied hard proposals.

    candidate_class:
        0 = NO_CANDIDATE
        1..num_colors = colors 0..num_colors-1
    """
    n = len(x)
    num_colors = x.shape[1] - 3

    if participate.shape != (n,) or candidate_class.shape != (n,):
        raise ValueError("A proposal per vertex is required")

    uncolored = x[:, 0] > 0.5

    participate = participate.to(
        device=x.device, dtype=x.dtype
    ) * uncolored

    candidate_class = candidate_class.to(
        device=x.device, dtype=torch.long
    )

    if bool(((candidate_class < 0) | (candidate_class > num_colors)).any()):
        raise ValueError(f"Candidate classes must be 0..{num_colors}")

    candidate_class = torch.where(
        uncolored,
        candidate_class,
        torch.zeros_like(candidate_class),
    )

    offer = F.one_hot(
        candidate_class, num_colors + 1
    ).to(dtype=x.dtype)

    phase2 = torch.cat(
        (x, participate[:, None], offer),
        dim=1,
    )

    messages = neighbor_max(
        model.message2(phase2),
        edge_index,
    )

    return model.commit_head(
        torch.cat((phase2, messages), dim=1)
    ).squeeze(-1)


def exact_commit_labels(
    x, edge_index, participate, candidate_class
):
    """
    Exact commit rule for the supplied proposals.

    Supports batches of disjoint graphs.
    """
    x = np.asarray(x)
    edges = np.asarray(edge_index)

    n = len(x)
    num_colors = x.shape[1] - 3
    part = np.asarray(participate, dtype=bool)
    cand = np.asarray(candidate_class, dtype=np.int64)

    if (
        part.shape != (n,)
        or cand.shape != (n,)
        or edges.shape[0] != 2
    ):
        raise ValueError("Inconsistent graph dimensions")

    if np.any((cand < 0) | (cand > num_colors)):
        raise ValueError(f"Candidate classes must be 0..{num_colors}")

    uncolored = x[:, 0] > 0.5

    # Color encoding matches candidate classes 1..num_colors.
    fixed_class = x[:, :num_colors + 1].argmax(axis=1)

    part = part & uncolored

    src, dst = edges

    # Conflict with a permanently colored neighbor.
    fixed_conflict = (
        (~uncolored[src])
        & (fixed_class[src] == cand[dst])
    )

    # Two adjacent participants propose the same color.
    simultaneous = (
        uncolored[src]
        & part[src]
        & (cand[src] == cand[dst])
        & (cand[dst] != 0)
    )

    blocked = np.zeros(n, dtype=bool)

    np.logical_or.at(
        blocked,
        dst,
        fixed_conflict | simultaneous,
    )

    return (
        uncolored
        & part
        & (cand != 0)
        & ~blocked
    ).astype(np.float32)


def make_counterfactual_proposals(
    x, edge_index, oracle_p, oracle_c, rng
):
    """
    Start from oracle proposals and deliberately
    inject invalid candidates and simultaneous
    conflicts.

    Does NOT modify the graph coloring.
    """
    x = np.asarray(x)
    edges = np.asarray(edge_index)

    num_colors = x.shape[1] - 3
    uncolored = x[:, 0] > 0.5
    fixed_class = x[:, :num_colors + 1].argmax(axis=1)

    part = np.asarray(
        oracle_p, dtype=bool
    ).copy()

    cand = np.asarray(
        oracle_c, dtype=np.int64
    ).copy()

    src, dst = edges

    # Inject conflicts with already-colored neighbors.
    fixed_neighbor = (
        (~uncolored[src]) & uncolored[dst]
    )

    for j in rng.permutation(
        np.flatnonzero(fixed_neighbor)
    ):
        if rng.random() < 0.17:
            v = int(dst[j])
            u = int(src[j])

            part[v] = True
            cand[v] = fixed_class[u]

    # Inject simultaneous candidate collisions.
    both_uncolored = (
        uncolored[src]
        & uncolored[dst]
        & (src < dst)
    )

    for j in rng.permutation(
        np.flatnonzero(both_uncolored)
    ):
        if rng.random() < 0.04:
            u = int(src[j])
            v = int(dst[j])

            color_class = int(
                rng.integers(1, num_colors + 1)
            )

            part[u] = True
            part[v] = True

            cand[u] = color_class
            cand[v] = color_class

    part[~uncolored] = False
    cand[~uncolored] = 0

    return part, cand
