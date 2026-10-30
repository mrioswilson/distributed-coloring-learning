#!/usr/bin/env python3
"""Analysis of protocol pedrorev-cf-v1 (Experiments A, B, C) on the frozen
confirmatory battery. Reads stored logits; no model is run or selected here.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from dcl.multiteacher.programs import KW
from dcl.pedrorev.evaluation import FAMILIES, be_commit_key
from dcl.study.storage import atomic_json, file_hash

DATA = Path("data/pedrorev-cf")
OUT = Path("results/pedro-revision/cf-experiment")


def confusion(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, tn = int((pred & truth).sum()), int((~pred & ~truth).sum())
    fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
    pos, neg = tp + fn, tn + fp
    tpr = tp / pos if pos else None
    tnr = tn / neg if neg else None
    return dict(n=int(len(truth)), tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / max(1, len(truth)), tpr=tpr, tnr=tnr,
                precision=tp / (tp + fp) if tp + fp else None,
                balanced_accuracy=(tpr + tnr) / 2 if tpr is not None and tnr is not None else None)


def strata(teacher, flags, bound):
    e, p, f, w = (flags & 1) > 0, (flags & 2) > 0, (flags & 4) > 0, (flags & 8) > 0
    if teacher == "BE_BASE_D3":
        return {"forced_ineligible": ~e & p, "silent": ~p, "eligible_active_free": e & p & f,
                "eligible_active_occupied": e & p & ~f,
                "late_eligible_active_occupied": e & p & ~f & (bound <= 12),
                "early_eligible_active_occupied": e & p & ~f & (bound > 12)}
    if teacher == "COLOR_TRIALS":
        return {"silent": ~p, "active_occupied": p & ~f, "active_free_nonwinning": p & f & ~w,
                "active_free_winning": p & f & w}
    return {"silent": ~p, "active_occupied": p & ~f, "active_free": p & f}


def rules(teacher, flags):
    e, p, f, w = (flags & 1) > 0, (flags & 2) > 0, (flags & 4) > 0, (flags & 8) > 0
    out = {"always_reject": np.zeros_like(p), "accept_iff_P": p, "accept_iff_P_and_F": p & f}
    if teacher == "BE_BASE_D3":
        out["accept_iff_P_and_E"] = p & e
        out["oracle_P_and_E_and_F"] = p & e & f
    if teacher == "COLOR_TRIALS":
        out["oracle_P_and_F_and_W"] = p & f & w
    if teacher == "SPECULATIVE_GM":
        out["oracle_P_and_F"] = p & f
    return out


def training_keys(bank_path, use_cf):
    with np.load(Path(bank_path) / "dataset.npz", allow_pickle=False) as f:
        t = {k: f[k] for k in f.files}
    rows = np.flatnonzero(t["native"][:, 3] == KW)
    x, rel, masks = t["a0"][rows], t["a2"][rows], t["a3"][rows]
    keys = set(be_commit_key(x, rel, masks, t["p"][rows], t["c"][rows]).tolist())
    if use_cf:
        for v in range(4):
            keys |= set(be_commit_key(x, rel, masks, t["cf_p"][rows, v], t["cf_c"][rows, v]).tolist())
    return keys


def type_metrics(keys, truth, pred, seen):
    order = np.argsort(keys, kind="stable")
    k, t, p = keys[order], truth[order], pred[order]
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    ends = np.r_[starts[1:], len(k)]
    rows, inconsistent = [], 0
    for a, b in zip(starts, ends):
        if p[a:b].min() != p[a:b].max() or t[a:b].min() != t[a:b].max():
            inconsistent += 1
        rows.append((int(k[a]), bool(t[a]), bool(p[a:b].mean() >= .5), b - a, int(k[a]) in seen))
    out = {}
    for name, sel in (("all", lambda r: True), ("seen", lambda r: r[4]), ("unseen", lambda r: not r[4])):
        rr = [r for r in rows if sel(r)]
        out[name] = dict(types=len(rr), occurrences=int(sum(r[3] for r in rr)),
                         **confusion([r[2] for r in rr], [r[1] for r in rr]))
    return out, inconsistent


def main():
    freeze = json.loads((OUT / "freeze.json").read_text())
    banks = {(b["teacher"], b["seed"], b["arm"]): b for b in json.loads((OUT / "banks.json").read_text())}
    results = []
    for teacher in freeze["config"]["teachers"]:
        qpath = DATA / "evaluation-banks" / teacher.lower() / "queries.npz"
        with np.load(qpath, allow_pickle=False) as raw:
            q = {k: raw[k] for k in raw.files}
        truth, flags, family, bound = q["y"] > .5, q["flags"], q["family"], q["bound"]
        st = strata(teacher, flags, bound)
        fam = {f: family == i for i, f in enumerate(FAMILIES)}
        for name, rule in rules(teacher, flags).items():
            for f, sel in fam.items():
                results.append(dict(teacher=teacher, arm=f"rule:{name}", seed=None, update=None, family=f,
                                    overall=confusion(rule[sel], truth[sel]),
                                    strata={k: confusion(rule[sel & s], truth[sel & s]) for k, s in st.items()}))
        for row in freeze["checkpoints"]:
            if row["teacher"] != teacher:
                continue
            path = DATA / "responses" / teacher.lower() / f"s{row['seed']}-{row['arm']}-u{row['update']:06d}.npy"
            pred = np.load(path) > 0
            for f, sel in fam.items():
                r = dict(teacher=teacher, arm=row["arm"], seed=row["seed"], update=row["update"], family=f,
                         labels=row["total_acquired_component_labels"], native_labels=row["native_labels"],
                         counterfactual_labels=row["counterfactual_labels"], checkpoint_sha256=row["sha256"],
                         responses_sha256=file_hash(path), overall=confusion(pred[sel], truth[sel]),
                         strata={k: confusion(pred[sel & s], truth[sel & s]) for k, s in st.items()})
                if teacher == "BE_BASE_D3" and f == "standard":
                    kw = sel & (q["be_key"] >= 0)
                    bank = banks[(teacher, row["seed"], row["arm"])]["bank"]
                    own = training_keys(bank, row["arm"].startswith("cf"))
                    shared = training_keys(banks[(teacher, row["seed"], "cf-high")]["bank"], True)
                    for label, seen in (("arm", own), ("shared_bank", shared)):
                        tm, bad = type_metrics(q["be_key"][kw], truth[kw], pred[kw], seen)
                        r[f"types_{label}"] = tm
                        r["within_type_inconsistent"] = bad
                        occ = np.isin(q["be_key"][kw], list(seen))
                        r[f"occurrences_{label}"] = dict(seen=confusion(pred[kw][occ], truth[kw][occ]),
                                                         unseen=confusion(pred[kw][~occ], truth[kw][~occ]))
                results.append(r)
    summary = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r["seed"] is None or r["update"] != 4000:
            continue
        key = summary[r["teacher"]][r["family"]].setdefault(r["arm"], defaultdict(list))
        key["balanced_accuracy"].append(r["overall"]["balanced_accuracy"])
        key["labels"].append(r["labels"])
        for s, c in r["strata"].items():
            key[f"{s}_tnr"].append(c["tnr"]); key[f"{s}_tpr"].append(c["tpr"])
        if "types_arm" in r:
            key["unseen_type_ba"].append(r["types_arm"]["unseen"]["balanced_accuracy"])
            key["unseen_types"].append(r["types_arm"]["unseen"]["types"])
            key["unseen_occurrence_ba"].append(r["occurrences_arm"]["unseen"]["balanced_accuracy"])
    atomic_json(OUT / "analysis.json", dict(rows=results, per_seed_summary=json.loads(json.dumps(summary)),
        script_sha256=file_hash(__file__), freeze_sha256=file_hash(OUT / "freeze.json")))
    for teacher, fams in summary.items():
        for f, arms in fams.items():
            for arm, d in arms.items():
                ba = [round(x * 100, 3) if x is not None else None for x in d["balanced_accuracy"]]
                print(teacher[:6], f[:8], f"{arm:14s}", "labels", d["labels"][0], "BA", ba)


if __name__ == "__main__":
    main()
