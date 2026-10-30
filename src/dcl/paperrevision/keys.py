"""Sufficient inputs to the queried-slot MAX COMMIT for BE and GM.

This is a sufficient, weight-independent equivalence, not a claim of a
minimal quotient. It follows RootedE1.second/read_commit and QueryPolicy's
relational adapter. It does not classify CANDIDATE or LINIAL/DETECT queries.
"""
import hashlib
import json
import numpy as np
from dcl.multiteacher.programs import KW, LINIAL, DETECT

SALT = "revision-20261030-commit-types-v1"


def commit_key(arrays, i, p, c, teacher):
    if teacher not in ("BE_BASE_D3", "SPECULATIVE_GM"):
        raise ValueError(teacher)
    x, _, rel, masks = arrays
    phase = int(x[i, 0, 8:14].argmax())
    if phase in (LINIAL, DETECT):
        raise ValueError("Not a continuation input")
    query = int(c[i, 0])
    be = teacher == "BE_BASE_D3"
    u = np.ones(4) if be else np.maximum(x[i, :, 0], x[i, :, 7])
    own = np.zeros(4) if be else x[i, :, 1 + query]
    root = (int(u[0]), int(own[0]), int(p[i, 0]), int(u[0] > .5))
    neighbors = set()
    for j in np.flatnonzero(masks[i, 0]):
        v = int(j) + 1
        occ = rel[i, 0, j, -4 + query] if phase == KW else own[v]
        neighbors.add((int(u[v]), int(occ), int(p[i, v]),
                       int(u[v] > .5 and c[i, v] == query),
                       0 if be else int(np.sign(x[i, v, 5] - x[i, 0, 5])),
                       0 if be else int(rel[i, 0, j, 1])))
    return (teacher, phase, int(x[i, 0, 7]),
            int(np.sign(x[i, 0, 18])) if phase == KW else 0,
            root, tuple(sorted(neighbors)))


def key_text(key):
    return json.dumps(key, separators=(",", ":"))


def held_out(key):
    return int.from_bytes(hashlib.sha256((SALT + key_text(key)).encode()).digest()[:8], "big") % 5 == 0
