#!/usr/bin/env python3
"""Re-analysis of the *existing* paired 4,000-update continuation diagnostics.

Reconstructs the original development queries (RNG 17001, as in
`scripts/final-counterfactual-audit.py`), reproduces every original numerator,
and adds what the earlier audit lacked: full confusion per seed and stratum,
sufficient-type (equal weight per distinct type) metrics with arm-specific and
shared-bank seen/unseen definitions, an empirical sufficiency check of the
type key, and privileged-predicate rule baselines. No training, no new
queries, no selection.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from dcl.multicf.oracles import proposal_bank
from dcl.multiteacher.programs import DETECT, KW, LINIAL
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.pedrorev.evaluation import annotate, be_commit_key
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash

BASE = Path("results/multiteacher-counterfactual")
OUT = Path("results/pedro-revision/existing-cf")
SEEDS = range(94101, 94106)
BUDGET = {"BE_BASE_D3": 3000, "SPECULATIVE_GM": 3000, "COLOR_TRIALS": 30000}


def confusion(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, tn = int((pred & truth).sum()), int((~pred & ~truth).sum())
    fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
    pos, neg = tp + fn, tn + fp
    tpr = tp / pos if pos else None
    tnr = tn / neg if neg else None
    return dict(n=len(truth), tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / max(1, len(truth)), tpr=tpr, tnr=tnr,
                precision=tp / (tp + fp) if tp + fp else None,
                balanced_accuracy=(tpr + tnr) / 2 if tpr is not None and tnr is not None else None)


def strata(teacher, flags):
    e, p, f, w = (flags & 1) > 0, (flags & 2) > 0, (flags & 4) > 0, (flags & 8) > 0
    if teacher == "BE_BASE_D3":
        return {"forced_ineligible": ~e & p, "silent": ~p, "eligible_active_free": e & p & f,
                "eligible_active_occupied": e & p & ~f}
    if teacher == "COLOR_TRIALS":
        return {"silent": ~p, "active_occupied": p & ~f, "active_free_nonwinning": p & f & ~w,
                "active_free_winning": p & f & w}
    return {"silent": ~p, "active_occupied": p & ~f, "active_free": p & f}


def rules(teacher, flags):
    e, p, f, w = (flags & 1) > 0, (flags & 2) > 0, (flags & 4) > 0, (flags & 8) > 0
    out = {"always_reject": np.zeros_like(p), "accept_iff_P": p}
    if teacher == "BE_BASE_D3":
        out["accept_iff_P_and_E"] = p & e
        out["accept_iff_P_and_F"] = p & f
    else:
        out["accept_iff_P_and_F"] = p & f
    if teacher == "COLOR_TRIALS":
        out["accept_iff_P_and_F_and_W"] = p & f & w
    return out


def training_keys(teacher, seed):
    directory = BASE / "datasets" / teacher.lower() / f"s{seed}" / f"b{BUDGET[teacher]}-coverage1-alternate0"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert file_hash(directory / "dataset.npz") == manifest["sha256"]
    with np.load(directory / "dataset.npz", allow_pickle=False) as f:
        t = {k: f[k] for k in f.files}
    rows = np.flatnonzero(t["native"][:, 3] == KW)
    x, rel, masks = t["a0"][rows], t["a2"][rows], t["a3"][rows]
    native = set(be_commit_key(x, rel, masks, t["p"][rows], t["c"][rows]).tolist())
    cf = set()
    for v in range(4):
        cf |= set(be_commit_key(x, rel, masks, t["cf_p"][rows, v], t["cf_c"][rows, v]).tolist())
    return native, native | cf, manifest["sha256"]


def per_type(keys, truth, pred, seen):
    """Equal weight per distinct sufficient type; decisions must be constant."""
    by = defaultdict(list)
    for i, k in enumerate(keys):
        by[int(k)].append(i)
    inconsistent, rows = 0, []
    for k, idx in by.items():
        d = pred[idx]
        if d.min() != d.max():
            inconsistent += 1
        rows.append((k, bool(truth[idx[0]]), bool(d.mean() >= .5), len(idx), k in seen))
    out = {}
    for name, sel in (("all", lambda r: True), ("seen", lambda r: r[4]), ("unseen", lambda r: not r[4])):
        rr = [r for r in rows if sel(r)]
        out[name] = dict(types=len(rr), **confusion([r[2] for r in rr], [r[1] for r in rr]),
                         occurrences=int(sum(r[3] for r in rr)))
    return out, inconsistent, rows


@torch.inference_mode()
def analyze(teacher):
    torch.set_num_threads(1)
    curves = json.loads((BASE / "training-curves.json").read_text())
    results = []
    for seed in SEEDS:
        recs = [r for r in curves if r["teacher"] == teacher and r["architecture"] == "normalized" and r["seed"] == seed
                and r["updates"] == 4000 and r["native_budget"] == BUDGET[teacher]
                and r["method"] in ("passive", "matched", "counterfactual")]
        assert {r["method"] for r in recs} == {"passive", "matched", "counterfactual"}
        models = {}
        for r in recs:
            assert file_hash(r["checkpoint"]) == r["checkpoint_sha256"]
            m = initialize(teacher, "normalized", seed).eval()
            m.load_state_dict(torch.load(r["checkpoint"], map_location="cpu", weights_only=False)["model"])
            models[r["method"]] = (m, r)
        bankdir = BASE / "diagnostics/common-state-banks" / teacher.lower() / f"s{seed}"
        bm = json.loads((bankdir / "manifest.json").read_text())
        assert file_hash(bankdir / "states.npz") == bm["sha256"]
        with np.load(bankdir / "states.npz", allow_pickle=False) as f:
            bank = {k: f[k] for k in f.files}
        rng = np.random.default_rng(17001)
        acc = defaultdict(list)
        for phase in sorted(np.unique(bank["native"][:, 3])):
            if phase in (LINIAL, DETECT):
                continue
            positions = np.flatnonzero(bank["native"][:, 3] == phase)
            for start in range(0, len(positions), 256):
                use = positions[start:start + 256]
                arrays = tuple(bank[f"a{i}"][use] for i in range(4))
                ps, cs, ys = proposal_bank(arrays, bank["native"][use], teacher, rng)
                states = {k: m.first(tensor_arrays(arrays), int(phase)) for k, (m, _) in models.items()}
                for v in range(4):
                    p, c = ps[:, v], cs[:, v]
                    flags, bound, key = annotate(arrays, teacher, p, c)
                    acc["truth"].append(ys[:, v] > .5); acc["flags"].append(flags); acc["key"].append(key)
                    acc["bound"].append(bound); acc["variant"].append(np.full(len(use), v))
                    for k, (m, _) in models.items():
                        logit = m.commit_for(states[k], torch.from_numpy(p), torch.from_numpy(c)).numpy()
                        acc[f"logit_{k}"].append(logit)
        A = {k: np.concatenate(v) for k, v in acc.items()}
        truth, flags = A["truth"], A["flags"]
        native_keys, bank_keys, bank_sha = training_keys(teacher, seed) if teacher == "BE_BASE_D3" else (set(), set(), None)
        for method, (m, rec) in models.items():
            old = json.loads((BASE / f"diagnostics/paired-u004000-{method}" / teacher.lower() / "normalized" / f"s{seed}.json").read_text())
            pred = A[f"logit_{method}"] > 0
            assert int((pred == truth).sum()) == old["components"]["counterfactual_correct"], (teacher, seed, method)
            row = dict(teacher=teacher, seed=seed, method=method, checkpoint_sha256=rec["checkpoint_sha256"],
                       labels=rec["total_acquired_component_labels"], overall=confusion(pred, truth),
                       strata={k: confusion(pred[s], truth[s]) for k, s in strata(teacher, flags).items()},
                       by_variant={str(v): confusion(pred[A["variant"] == v], truth[A["variant"] == v]) for v in range(4)},
                       reproduces_original_numerator=True)
            if teacher == "BE_BASE_D3":
                own_seen = native_keys if method != "counterfactual" else bank_keys
                for label, seen in (("arm_seen", own_seen), ("bank_seen", bank_keys)):
                    types, bad, _ = per_type(A["key"], truth, pred, seen)
                    row[f"types_{label}"] = types
                    row["within_type_inconsistent"] = bad
                    occ_seen = np.isin(A["key"], list(seen))
                    row[f"occurrences_{label}"] = dict(seen=confusion(pred[occ_seen], truth[occ_seen]),
                                                      unseen=confusion(pred[~occ_seen], truth[~occ_seen]))
                late = (A["bound"] <= 12) & ((flags & 7) == 3)
                row["late_eligible_active_occupied"] = confusion(pred[late], truth[late])
                row["max_within_type_logit_range"] = float(max(
                    (np.ptp(A[f"logit_{method}"][A["key"] == k]) for k in np.unique(A["key"])), default=0.))
            results.append(row)
        for name, rule in rules(teacher, flags).items():
            row = dict(teacher=teacher, seed=seed, method=f"rule:{name}", overall=confusion(rule, truth),
                       strata={k: confusion(rule[s], truth[s]) for k, s in strata(teacher, flags).items()})
            if teacher == "BE_BASE_D3":
                row["types_bank_seen"], _, _ = per_type(A["key"], truth, rule.astype(float), bank_keys)
            results.append(row)
        print(teacher, seed, {r["method"]: round(r["overall"]["balanced_accuracy"], 6) for r in results if r["seed"] == seed},
              flush=True)
    return results


def main():
    rows = []
    for teacher in ("BE_BASE_D3", "SPECULATIVE_GM", "COLOR_TRIALS"):
        rows += analyze(teacher)
    summary = defaultdict(dict)
    for teacher in ("BE_BASE_D3", "SPECULATIVE_GM", "COLOR_TRIALS"):
        for method in sorted({r["method"] for r in rows if r["teacher"] == teacher}):
            rr = [r for r in rows if r["teacher"] == teacher and r["method"] == method]
            summary[teacher][method] = dict(
                mean_accuracy=float(np.mean([r["overall"]["accuracy"] for r in rr])),
                mean_balanced_accuracy=float(np.mean([r["overall"]["balanced_accuracy"] for r in rr])),
                min_balanced_accuracy=float(np.min([r["overall"]["balanced_accuracy"] for r in rr])),
                max_balanced_accuracy=float(np.max([r["overall"]["balanced_accuracy"] for r in rr])))
    atomic_json(OUT / "existing-cf-reanalysis.json", dict(rows=rows, summary=summary, script_sha256=file_hash(__file__),
        scope="Existing development diagnostics (60 trajectories n28/36 per seed, RNG 17001); reused development split; "
              "rules read privileged decoded predicates and are not trained competitors."))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
