"""Alternative candidate rule, with the unchanged participation/COMMIT rules."""
import numpy as np

from dcl.universal.rules import available_matrix, exact_commit_labels, teacher_offer


def cyclic_offer(colors, rc, edge_index, palette):
    colors = np.asarray(colors)
    palette = np.broadcast_to(np.asarray(palette), colors.shape)
    available = available_matrix(colors, edge_index, palette)
    starts = np.floor(np.asarray(rc) * palette).astype(np.int64)
    if np.any(starts < 0) or np.any(starts >= palette):
        raise ValueError("Uniform draws must belong to [0,1)")
    result = np.full(len(colors), -1, dtype=np.int64)
    for v in np.flatnonzero((colors < 0) & available.any(1)):
        order = (starts[v] + np.arange(palette[v])) % palette[v]
        result[v] = order[np.flatnonzero(available[v, order])[0]]
    return result, available.sum(1)


def cyclic_law(available):
    available = np.asarray(available, bool)
    indices = np.flatnonzero(available)
    law = np.zeros(len(available), float)
    if len(indices):
        gaps = (indices - np.roll(indices, 1)) % len(available)
        if len(indices) == 1:
            gaps[0] = len(available)
        law[indices] = gaps / len(available)
    return law


def teacher_trajectory(n, edges, palette, seed, cyclic=False, max_rounds=1000):
    from dcl.universal.data import UExample
    colors = np.full(n, -1, np.int64)
    rng, examples = np.random.default_rng(seed), []
    rule = cyclic_offer if cyclic else teacher_offer
    for _ in range(max_rounds):
        rp, rc = rng.random(n), rng.random(n)
        candidates, k = rule(colors, rc, edges, palette)
        part = (colors < 0) & (k > 0) & (rp < .5)
        commit = exact_commit_labels(colors, edges, part, candidates)
        examples.append(UExample(colors.copy(), rp.astype(np.float32), rc.astype(np.float32), edges,
                                 palette, palette-1, part.astype(np.float32), candidates, commit.astype(np.float32)))
        colors[commit] = candidates[commit]
        if np.all(colors >= 0):
            return examples
    raise RuntimeError("Teacher did not finish within its budget")
