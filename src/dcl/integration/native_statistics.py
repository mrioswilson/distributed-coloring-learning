"""Audited V2 with an explicitly declared common palette alphabet.

The historical feature allocator assumes K=max(degree)+1. Native teachers
also use color3 on paths of degree2. An isolated import reuses every V2
selection/permutation function, replacing only that histogram allocation.
Actual degrees and graph edges are never changed. No shared module is patched.
"""
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np

from dcl.robustdist import stats as historical

_spec = spec_from_file_location("dcl.robustdist._declared_palette_v2", historical.__file__)
_v2 = module_from_spec(_spec)
_spec.loader.exec_module(_v2)


def structural(z, g):
    k = g.get("palette_size", max(g["degrees"]) + 1)
    if k == max(g["degrees"]) + 1:
        return historical.structural(z, g)
    if k < 1 or np.any((z < 0) | (z >= k)):
        raise ValueError("Completed colors must belong to the declared alphabet")
    n = g["n"]
    counts = np.zeros((len(z), k), float)
    for i, row in enumerate(z):
        counts[i] = np.bincount(row, minlength=k)
    used = (counts > 0).sum(1); p = counts / n
    features = [used, counts.max(1)/n, -(p*np.log(np.maximum(p, 1e-100))).sum(1), z.mean(1)]
    names = ["used_colors", "largest_class_fraction", "class_entropy", "mean_color"]
    pairs, _ = historical.groups(g)
    edge = set(map(tuple, np.array(g["edges"], int).reshape(2, -1).T.tolist()))
    for a, b in pairs:
        if (a, b) not in edge:
            features.append((z[:, a] == z[:, b]).astype(float)); names.append(f"cocolor:{a},{b}")
    for degree in sorted(set(g["degrees"])):
        ids = np.flatnonzero(np.array(g["degrees"]) == degree)
        features.extend([z[:, ids].mean(1), np.sum(z[:, ids] % 2, axis=1) % 2])
        names.extend([f"degree_mean:{degree}", f"degree_parity:{degree}"])
    return np.column_stack(features), names


_v2.structural = structural
joint_test = _v2.joint_test
