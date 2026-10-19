"""Exact protocol rules for labels and read-only diagnostics.

Nothing in this module is called by the model or by the action of a learned
rollout. Training uses it to build COMMIT labels for the proposals actually
given to the COMMIT stage; evaluation uses it after the action is fixed.
Colors and candidates use -1 for "uncolored" / "no candidate".
"""
from __future__ import annotations

import numpy as np


def available_matrix(colors, edge_index, palette):
    """available[v, c]: c < palette[v] and no fixed neighbor of v has color c."""
    colors = np.asarray(colors)
    palette = np.broadcast_to(np.asarray(palette), colors.shape)
    k = int(palette.max())
    available = np.arange(k)[None, :] < palette[:, None]
    src, dst = np.asarray(edge_index)
    fixed = colors[src] >= 0
    available[dst[fixed], colors[src[fixed]]] = False
    return available


def teacher_offer(colors, rc, edge_index, palette):
    """Color the teacher would propose: A(v)[floor(r_c * |A(v)|)], or -1."""
    colors = np.asarray(colors)
    available = available_matrix(colors, edge_index, palette)
    k = available.sum(1)
    rank = np.floor(np.asarray(rc) * k).astype(np.int64)
    position = available.cumsum(1) - 1
    offer = (available & (position == rank[:, None])).argmax(1).astype(np.int64)
    offer[(colors >= 0) | (k == 0)] = -1
    return offer, k


def teacher_participation(colors, rp, palette_nonempty=None, probability=0.5):
    part = (np.asarray(colors) < 0) & (np.asarray(rp) < probability)
    if palette_nonempty is not None:
        part &= palette_nonempty
    return part


def conflict_flags(colors, edge_index, participate, candidate):
    """Per node: (fixed-neighbor conflict, simultaneous conflict) of its candidate."""
    colors = np.asarray(colors)
    uncolored = colors < 0
    part = np.asarray(participate, dtype=bool) & uncolored
    cand = np.where(uncolored, np.asarray(candidate, dtype=np.int64), -1)
    src, dst = np.asarray(edge_index)
    fixed = (~uncolored[src]) & (colors[src] == cand[dst]) & (cand[dst] >= 0)
    simultaneous = uncolored[src] & part[src] & (cand[src] == cand[dst]) & (cand[dst] >= 0)
    fixed_conflict = np.zeros(len(colors), dtype=bool)
    simultaneous_conflict = np.zeros(len(colors), dtype=bool)
    np.logical_or.at(fixed_conflict, dst, fixed)
    np.logical_or.at(simultaneous_conflict, dst, simultaneous)
    return fixed_conflict, simultaneous_conflict


def exact_commit_labels(colors, edge_index, participate, candidate):
    """COMMIT is safe iff the node participates with a candidate that no fixed
    neighbor holds and no participating uncolored neighbor also proposes."""
    colors = np.asarray(colors)
    uncolored = colors < 0
    part = np.asarray(participate, dtype=bool) & uncolored
    cand = np.where(uncolored, np.asarray(candidate, dtype=np.int64), -1)
    fixed, simultaneous = conflict_flags(colors, edge_index, part, cand)
    return uncolored & part & (cand >= 0) & ~fixed & ~simultaneous


def make_counterfactual_proposals(colors, edge_index, palette, oracle_p, oracle_c, rng,
                                  fixed_probability=0.17, simultaneous_probability=0.04):
    """M6B counterfactual proposals for arbitrary palettes.

    Same procedure as dcl.counterfactual.make_counterfactual_proposals: start
    from the teacher's proposals, then (1) for each edge from a fixed neighbor u
    to an uncolored v, with probability 0.17 make v participate proposing
    color(u); (2) for each edge between uncolored nodes, with probability 0.04
    make both participate proposing the same uniformly drawn palette color.
    Edges are visited in a random order and later injections overwrite.
    The coloring is never modified.
    """
    colors = np.asarray(colors)
    palette = np.broadcast_to(np.asarray(palette), colors.shape)
    uncolored = colors < 0
    part = np.asarray(oracle_p, dtype=bool).copy()
    cand = np.asarray(oracle_c, dtype=np.int64).copy()
    src, dst = np.asarray(edge_index)
    eligible = rng.permutation(np.flatnonzero((~uncolored[src]) & uncolored[dst]))
    for j in eligible[rng.random(len(eligible)) < fixed_probability]:
        part[dst[j]] = True
        cand[dst[j]] = colors[src[j]]
    eligible = rng.permutation(np.flatnonzero(uncolored[src] & uncolored[dst] & (src < dst)))
    chosen = eligible[rng.random(len(eligible)) < simultaneous_probability]
    drawn = rng.random(len(chosen))
    for j, r in zip(chosen, drawn):
        u, v = src[j], dst[j]
        color = int(r * palette[u])
        part[u] = part[v] = True
        cand[u] = cand[v] = color
    part[~uncolored] = False
    cand[~uncolored] = -1
    return part, cand
