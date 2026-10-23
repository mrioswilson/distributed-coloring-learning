import numpy as np
import torch

from dcl.behavior_statistics import kernel_test, canonical_colors
from dcl.robustdist.stats import joint_test
from dcl.integration.native_statistics import joint_test as native_joint_test


def test_pedro_signed_statistic_equals_independent_unbiased_mmd_with_status():
    torch.set_num_threads(1)
    x = np.array([[0, 1], [1, 0], [0, 2]])
    y = np.array([[1, 2], [2, 1], [2, 0]])
    sx, sy = np.array([0, 0, 2]), np.array([0, 2, 2])
    scales = (1., 4., 16.)
    def kernel(a, b, sa, sb):
        return float(sa == sb) * np.mean([np.exp(-s * np.mean(a != b)) for s in scales])
    within = sum(kernel(x[i], x[j], sx[i], sx[j]) + kernel(y[i], y[j], sy[i], sy[j])
                 for i in range(3) for j in range(3) if i != j) / 6
    cross = sum(kernel(a, b, sa, sb) for a, sa in zip(x, sx) for b, sb in zip(y, sy)) / 9
    result = kernel_test(x, y, sx, sy, dict(kernel_scales=scales, permutations=19), np.random.default_rng(1))
    assert abs(result["mmd2"] - (within - 2 * cross)) < 1e-14


def test_both_statistics_conservatively_keep_exact_point_mass_null():
    x = np.tile([0, 1], (8, 1)); status = np.zeros(8, int)
    old = kernel_test(x, x, status, status, dict(kernel_scales=[1., 4., 16.], permutations=99), np.random.default_rng(2))
    g = dict(id="audit-edge", n=2, degrees=[1, 1], edges=[[0, 1], [1, 0]])
    new = joint_test(x, x, g, np.random.default_rng(3), B=99)
    assert old["p_value"] == new["p"] == 1.


def test_color_quotient_can_hide_completely_disjoint_labeled_laws():
    a, b = np.array([0, 1, 0]), np.array([2, 0, 2])
    assert not np.array_equal(a, b)
    assert np.array_equal(canonical_colors(a), canonical_colors(b))


def test_native_global_four_color_path_has_valid_point_null_without_changing_degrees():
    x = np.tile([3, 0, 3], (8, 1))
    g = dict(id="native-path", n=3, degrees=[1, 2, 1], edges=[[0, 1, 1, 2], [1, 0, 2, 1]], palette_size=4)
    r = native_joint_test(x, x, g, np.random.default_rng(4), B=99)
    assert r["p"] == 1. and g["degrees"] == [1, 2, 1]


def test_declared_palette_adapter_reproduces_original_v2_when_alphabet_already_matches():
    import itertools
    support = np.array(list(itertools.permutations(range(4))))
    rng = np.random.default_rng(5)
    x, y = support[rng.integers(24, size=16)], support[rng.integers(24, size=16)]
    e = [(a, b) for a in range(4) for b in range(4) if a != b]
    g = dict(id="audit-k4", n=4, degrees=[3]*4, edges=np.array(e).T.tolist(), palette_size=4)
    a = joint_test(x, y, g, np.random.default_rng(6), B=199)
    b = native_joint_test(x, y, g, np.random.default_rng(6), B=199)
    assert a["p"] == b["p"] and a["statistics"] == b["statistics"]
