"""Statistics shared by behavior studies, independent of any neural model.

These functions preserve the definitions used in luby_behavior.py. A two-sample
test detects differences; only the coupling bound can certify the specified TV
tolerance, and only for the fixed graph and frozen model under consideration.
"""
from collections import Counter
import csv
import io

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta
import torch

from dcl.study.storage import atomic_write


def canonical_colors(colors):
    """Quotient by global color permutations, keeping vertex labels and -1."""
    rows = np.atleast_2d(colors)
    result = np.full_like(rows, -1)
    for i, row in enumerate(rows):
        names = {}
        for j, color in enumerate(row):
            if color >= 0:
                names.setdefault(int(color), len(names))
                result[i, j] = names[int(color)]
    return result if np.ndim(colors) == 2 else result[0]


def empirical_tv(a, b):
    """Descriptive plug-in distance; not a confidence bound on population TV."""
    ca, cb = Counter(map(tuple, a)), Counter(map(tuple, b))
    return sum(abs(ca[key] / len(a) - cb[key] / len(b)) for key in ca.keys() | cb.keys()) / 2


def kernel_test(a, b, status_a, status_b, plan, rng):
    """Unbiased MMD², characteristic Hamming mixture × terminal-status delta.

    Balanced permutations are made within a fixed graph, using independent
    samples A and B. The returned null draws also support stratified tests.
    """
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2:
        raise ValueError("Two equally sized coloring samples required")
    m = len(a)
    colors, status = np.concatenate((a, b)), np.concatenate((status_a, status_b))
    distance = cdist(colors, colors, metric="hamming")
    kernel = sum(np.exp(-s * distance) for s in plan["kernel_scales"]) / len(plan["kernel_scales"])
    kernel *= status[:, None] == status[None, :]
    np.fill_diagonal(kernel, 0)
    signs = np.empty((plan["permutations"] + 1, 2 * m), np.float64)
    signs[0] = np.concatenate((-np.ones(m), np.ones(m)))
    for i in range(1, len(signs)):
        signs[i] = signs[0, rng.permutation(2 * m)]
    product = (torch.from_numpy(signs) @ torch.from_numpy(kernel)).numpy()
    signed = np.einsum("ij,ij->i", product, signs)
    values = (kernel.sum() + (2 * m - 1) * signed) / (2 * m * m * (m - 1))
    p = (1 + np.count_nonzero(values[1:] >= values[0] - 1e-14)) / len(values)
    return {"mmd2": float(values[0]), "p_value": float(p),
            "null95": float(np.quantile(values[1:], 0.95)), "null": values[1:]}


def coupling_bound(errors, samples, alpha=0.05):
    """One-sided exact binomial bound on final disagreement, hence on TV."""
    if not 0 <= errors <= samples or samples < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid binomial inputs")
    return 1.0 if errors == samples else float(beta.ppf(1 - alpha, errors + 1, samples - errors))


def holm_adjust(p_values):
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    result = np.empty(len(order))
    previous = 0.0
    for i, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - i) * p_values[index]))
        result[index] = previous
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, lambda f: f.write(stream.getvalue().encode()))
