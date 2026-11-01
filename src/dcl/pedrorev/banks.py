"""Label-budget-controlled and held-out-family training banks.

Every bank is derived from an immutable `dcl.multicf.data.acquire` bank. A
derived bank only *removes* acquired continuation labels: a removed query is
replaced by the root's native (teacher-message) query and native label, which
the native bank already contains, so it is never charged as a new label.
Nothing here is imported by a policy or by autonomous inference.
"""
import json
from pathlib import Path

import numpy as np

from dcl.multicf.data import acquire, load
from dcl.multicf.oracles import confirmation
from dcl.multiteacher.programs import DETECT, KW, LINIAL, named_seed
from dcl.study.storage import atomic_json, file_hash

ID_SPACE = 65536


def be_query_flags(a0, a2, a3, p, c):
    """E/P/F flags of a BE KW rooted query, from the stored encoded inputs.

    E: root is KW-eligible (old auxiliary color >= target, i.e. must recolor).
    P: root participates in the query. F: queried option is not the current
    auxiliary color of any neighbor (receiver-relative option equality).
    """
    eligible = a0[:, 0, 18] >= 0
    participates = p[:, 0].astype(bool)
    occupied = ((a2[:, 0, :, -4:] > .5) & a3[:, 0, :, None]).any(1)
    free = ~occupied[np.arange(len(c)), c[:, 0]]
    bound = np.rint(a0[:, 0, 17] * ID_SPACE).astype(np.int64)
    return eligible, participates, free, bound


def bank_dir(root, teacher, seed, budget):
    return Path(root) / "acquired" / teacher.lower() / f"s{seed}" / f"b{budget}-coverage1-alternate0"


def acquire_native(config, teacher, seed, budget, root):
    directory = bank_dir(root, teacher, seed, budget)
    manifest = acquire(config, teacher, seed, budget, directory, True, False)
    return directory, manifest


def _write_derived(directory, packed, base_manifest, base_dir, modification, kept_cf):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "dataset.npz"
    if not target.exists():
        np.savez_compressed(target, **packed)
    manifest = dict(identity=dict(base_manifest["identity"], derived=modification["kind"]),
        derived_from=str(base_dir), derived_from_sha256=base_manifest["sha256"],
        native_labels=base_manifest["native_labels"], counterfactual_labels=int(kept_cf),
        total_acquired_component_labels=base_manifest["native_labels"] + int(kept_cf),
        modification=modification, sha256=file_hash(target), examples=len(packed["native"]),
        accounting="Removed continuation queries are replaced by the root's native teacher-message query "
                   "and native label already present in the native bank; they are not charged.")
    path = directory / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["sha256"] != manifest["sha256"] or old["modification"] != modification:
            raise RuntimeError("Existing derived bank differs")
        return old
    atomic_json(path, manifest)
    return manifest


def _replace_with_native(packed, rows, variants, teacher):
    """Replace selected (row, variant) continuation queries by native queries."""
    cf_p, cf_c, cf_y = packed["cf_p"].copy(), packed["cf_c"].copy(), packed["cf_y"].copy()
    cf_p[rows, variants] = packed["p"][rows]
    cf_c[rows, variants] = packed["c"][rows]
    cf_y[rows, variants] = (packed["native"][rows, 0] > 0).astype(np.float32)
    out = dict(packed, cf_p=cf_p, cf_c=cf_c, cf_y=cf_y)
    # Re-verify every replaced label with the declared oracle.
    for phase in np.unique(packed["native"][rows, 3]):
        use = rows[packed["native"][rows, 3] == phase]
        vv = variants[packed["native"][rows, 3] == phase]
        arrays = tuple(packed[f"a{i}"][use] for i in range(4))
        y = confirmation(arrays, teacher, cf_p[use, vv], cf_c[use, vv])
        if not np.array_equal(y, cf_y[use, vv]):
            raise RuntimeError("Native replacement label differs from oracle")
    return out


def color_rows(packed):
    return np.flatnonzero(~np.isin(packed["native"][:, 3], (LINIAL, DETECT)))


def truncated_cf_bank(config, teacher, seed, total, root):
    """Smallest native budget whose native+CF labels reach `total`, then drop
    randomly chosen continuation queries until exactly `total` are charged."""
    probe = max(60, int(total / 2.4))
    while True:
        base_dir, base = acquire_native(config, teacher, seed, probe, root)
        if base["native_labels"] + base["counterfactual_labels"] >= total:
            break
        probe = int(probe * 1.05) + 5
    packed, _ = load(base_dir)
    keep = total - base["native_labels"]
    rows = color_rows(packed)
    pairs = np.array([(r, v) for r in rows for v in range(4)], dtype=np.int64)
    rng = np.random.default_rng(named_seed(config["namespace"], "truncate", teacher, seed, total))
    drop = rng.permutation(len(pairs))[:len(pairs) - keep]
    derived = _replace_with_native(packed, pairs[drop, 0], pairs[drop, 1], teacher) if len(drop) else packed
    modification = dict(kind=f"cf-total-{total}", base_native_budget=probe, dropped_queries=int(len(drop)),
                        kept_queries=int(keep), rule="uniform random (row,variant) removal, fixed RNG")
    directory = Path(root) / "derived" / teacher.lower() / f"s{seed}" / f"cf-total-{total}"
    return directory, _write_derived(directory, derived, base, base_dir, modification, keep)


def holdout_bank(config, teacher, seed, budget, kind, root):
    """BE held-out continuation families (Experiment B).

    noforce: remove every query in which a KW-ineligible root is made to
      participate (forced-participation family; label 0 by contract).
    late-occupied: remove every eligible, participating query whose option is
      occupied at late KW steps (bound <= 12): a held-out *combination* of an
      intervention type seen early with phase values seen only natively.
    """
    if teacher != "BE_BASE_D3":
        raise ValueError("Held-out families are defined for BE only")
    base_dir, base = acquire_native(config, teacher, seed, budget, root)
    packed, _ = load(base_dir)
    rows = np.flatnonzero(packed["native"][:, 3] == KW)
    drop_r, drop_v = [], []
    for v in range(4):
        e, p, f, bound = be_query_flags(packed["a0"][rows], packed["a2"][rows], packed["a3"][rows],
                                        packed["cf_p"][rows, v], packed["cf_c"][rows, v])
        if kind == "noforce":
            hit = ~e & p
        elif kind == "late-occupied":
            hit = e & p & ~f & (bound <= 12)
        else:
            raise ValueError(kind)
        drop_r.append(rows[hit]); drop_v.append(np.full(int(hit.sum()), v))
    drop_r, drop_v = np.concatenate(drop_r), np.concatenate(drop_v)
    derived = _replace_with_native(packed, drop_r, drop_v, teacher)
    keep = base["counterfactual_labels"] - len(drop_r)
    modification = dict(kind=f"holdout-{kind}", base_native_budget=budget, dropped_queries=int(len(drop_r)),
                        kept_queries=int(keep))
    directory = Path(root) / "derived" / teacher.lower() / f"s{seed}" / f"holdout-{kind}"
    return directory, _write_derived(directory, derived, base, base_dir, modification, keep)


def neighbor_bank(config, teacher, seed, budget, root):
    """Experiment F: the cf-high bank with neighbor messages intervened.

    Same roots, root queries and acquired labels; only the received neighbor
    messages of every continuation query are perturbed (Luby-style generator:
    candidates resampled w.p. .25, permitted participation flipped w.p. .15,
    variant 1 copies the root candidate to all permitted neighbors). For BE
    the contract label does not read neighbor messages; this is verified.
    """
    base_dir, base = acquire_native(config, teacher, seed, budget, root)
    packed, _ = load(base_dir)
    rng = np.random.default_rng(named_seed(config["namespace"], "neighbor-interventions", teacher, seed))
    cf_p, cf_c = packed["cf_p"].copy(), packed["cf_c"].copy()
    rows = color_rows(packed)
    allowed = packed["a0"][rows][..., 7] > .5
    for v in range(4):
        p, c = cf_p[rows, v], cf_c[rows, v]
        b, centers = p.shape
        change = rng.random((b, centers - 1)) < .25
        c[:, 1:] = np.where(change, rng.integers(4, size=(b, centers - 1)), c[:, 1:])
        p[:, 1:] ^= (rng.random((b, centers - 1)) < .15) & allowed[:, 1:]
        if v == 1:
            c[:, 1:] = c[:, :1]
            p[:, 1:] = allowed[:, 1:]
        p &= allowed
        cf_p[rows, v], cf_c[rows, v] = p, c
    for phase in np.unique(packed["native"][rows, 3]):
        use = rows[packed["native"][rows, 3] == phase]
        arrays = tuple(packed[f"a{i}"][use] for i in range(4))
        for v in range(4):
            y = confirmation(arrays, teacher, cf_p[use, v], cf_c[use, v])
            if not np.array_equal(y, packed["cf_y"][use, v]):
                raise RuntimeError("Neighbor intervention changed a contract label")
    derived = dict(packed, cf_p=cf_p, cf_c=cf_c)
    modification = dict(kind="neighbor-interventions", base_native_budget=budget, dropped_queries=0,
                        kept_queries=int(base["counterfactual_labels"]),
                        rule="Luby-style neighbor message perturbation of every stored continuation query; labels unchanged")
    directory = Path(root) / "derived" / teacher.lower() / f"s{seed}" / "neighbor-interventions"
    return directory, _write_derived(directory, derived, base, base_dir, modification, base["counterfactual_labels"])
