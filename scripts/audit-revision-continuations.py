#!/usr/bin/env python3
"""Post-test diagnostics from frozen banks; never trains or rewrites the trial."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from dcl.multiteacher.programs import ASSIGN, DETECT, KW, LINIAL, REPAIR
from dcl.paperrevision.keys import commit_key, held_out, key_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper-revision-20261030/continuation-audit.json"
PHASES = {KW: "KW", ASSIGN: "ASSIGN", REPAIR: "REPAIR"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as bank:
        return {key: bank[key] for key in bank.files}


def facts(bank, teacher, prefix=""):
    """Recompute the logical oracle directly from encoded raw state arrays."""
    x, rel, mask = bank["a0"], bank["a2"], bank["a3"]
    p, c = bank[prefix + "p"], bank[prefix + "c"]
    occupancy = rel[:, 0, :, -4:] > .5 if teacher == "BE_BASE_D3" else bank["a1"][:, 0, :, 1:5] > .5
    occupied = (occupancy & mask[:, 0, :, None]).any(axis=1)[np.arange(len(x)), c[:, 0]]
    eligible = x[:, 0, 7] > .5
    if teacher == "BE_BASE_D3":
        eligible &= x[:, 0, 18] >= 0
    active = p[:, 0].astype(bool)
    truth = eligible & active & ~occupied
    logical = [(int(ph), bool(e), bool(a), bool(o))
               for ph, e, a, o in zip(bank["native"][:, 3], eligible, active, occupied)]
    arrays = tuple(bank[f"a{j}"] for j in range(4))
    fine = [key_text(commit_key(arrays, i, p, c, teacher)) for i in range(len(x))]
    return dict(eligible=eligible, active=active, occupied=occupied,
                truth=truth, logical=logical, fine=fine)


def confusion(pred, truth):
    tp, tn = int((pred & truth).sum()), int((~pred & ~truth).sum())
    fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, n=len(truth),
                accuracy=(tp + tn) / len(truth),
                balanced_accuracy=.5 * (tp / (tp + fn) + tn / (tn + fp))
                if tp + fn and tn + fp else None)


def labels(truth):
    return dict(positive=int(truth.sum()), negative=int((~truth).sum()))


def plus_labels(left, right):
    return {k: left[k] + right[k] for k in ("positive", "negative")}


def bounds(values):
    return [min(values), max(values)]


def inspect_pair(runtime, teacher, seed, input_hashes):
    data = runtime / "datasets" / teacher.lower() / f"s{seed}"
    testdir = runtime / "test" / teacher.lower() / f"s{seed}"
    paths = [data / "base.npz", data / "aux.npz", testdir / "aux.npz",
             data / "type-keys.json", testdir / "type-keys.json"]
    base, aux, test = [load(p) for p in paths[:3]]
    use = ~np.isin(base["native"][:, 3], [LINIAL, DETECT])
    base = {k: v[use] for k, v in base.items()}
    bf, nf, cf, tf = (facts(b, teacher, prefix) for b, prefix in
                      ((base, ""), (aux, ""), (aux, "cf_"), (test, "cf_")))
    assert np.array_equal(bf["truth"], base["native"][:, 0] > 0)
    assert np.array_equal(nf["truth"], aux["native_y"] > .5)
    assert np.array_equal(cf["truth"], aux["cf_y"] > .5)
    assert np.array_equal(tf["truth"], test["cf_y"] > .5)
    train_fine = set(bf["fine"]) | set(nf["fine"]) | set(cf["fine"])
    assert train_fine == set(json.loads(paths[3].read_text()))
    assert set(tf["fine"]) == set(json.loads(paths[4].read_text()))
    assert not train_fine.intersection(tf["fine"])
    assert all(not held_out(json.loads(k)) for k in train_fine)
    assert all(held_out(json.loads(k)) for k in set(tf["fine"]))

    truth = tf["truth"]
    gate = tf["eligible"] & tf["active"]
    gate_metrics = confusion(gate, truth)
    by_type = defaultdict(list)
    type_truth = {}
    for key, good, y in zip(tf["fine"], gate == truth, truth):
        by_type[key].append(bool(good))
        if key in type_truth:
            assert type_truth[key] == bool(y)
        type_truth[key] = bool(y)
    gate_metrics["equal_type_accuracy"] = float(np.mean([np.mean(v) for v in by_type.values()]))

    phase_labels = {}
    for phase in sorted(np.unique(aux["native"][:, 3])):
        b = base["native"][:, 3] == phase
        a = aux["native"][:, 3] == phase
        phase_labels[PHASES[int(phase)]] = {
            arm: plus_labels(labels(bf["truth"][b]), labels(f["truth"][a]))
            for arm, f in (("native", nf), ("cf", cf))}
    train_logical = {arm: set(bf["logical"]) | set(f["logical"])
                     for arm, f in (("native", nf), ("cf", cf))}
    model_metrics = {}
    for arm in ("native", "cf"):
        path = ROOT / "results/paper-revision-20261030/prospective" / teacher.lower() / f"s{seed}" / (arm + "-test.json")
        paths.append(path)
        diagnostic = json.loads(path.read_text())["diagnostic"]
        assert diagnostic["n"] == len(truth)
        assert diagnostic["tp"] + diagnostic["fn"] == int(truth.sum())
        assert diagnostic["distinct_types"] == len(by_type)
        model_metrics[arm] = diagnostic
    strata = {}
    for name, sel in (("eligible_active", gate),
                      ("eligible_silent", tf["eligible"] & ~tf["active"]),
                      ("ineligible", ~tf["eligible"])):
        if sel.any():
            strata[name] = dict(truth=labels(truth[sel]), gate=confusion(gate[sel], truth[sel]))
            for arm in ("native", "cf"):
                strata[name][arm] = model_metrics[arm]["strata"][name]
                assert strata[name][arm]["n"] == int(sel.sum())
                assert strata[name][arm]["tp"] + strata[name][arm]["fn"] == int(truth[sel].sum())
    for path in paths:
        input_hashes[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = digest(path)
    counts = Counter(tf["fine"])
    return dict(teacher=teacher, seed=seed, queries=len(truth),
                train_color_labels={arm: plus_labels(labels(bf["truth"]), labels(f["truth"]))
                                    for arm, f in (("native", nf), ("cf", cf))},
                train_labels_by_phase=phase_labels, test_labels=labels(truth),
                sufficient_test_types=len(counts), max_fine_type_fraction=max(counts.values()) / len(truth),
                logical_test_types=len(set(tf["logical"])),
                logical_test_occurrence_overlap={arm: float(np.mean([k in keys for k in tf["logical"]]))
                                                  for arm, keys in train_logical.items()},
                logical_test_type_overlap={arm: len(set(tf["logical"]) & keys) for arm, keys in train_logical.items()},
                fine_train_test_intersection=0, gate=gate_metrics,
                model_metrics=model_metrics, strata=strata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=ROOT / "data/paper-revision-20261030")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/paper-revision-20261030.json").read_text())
    rows, hashes, summary = [], {}, {}
    for teacher in config["teachers"]:
        for seed in config["seeds"]:
            rows.append(inspect_pair(args.runtime.resolve(), teacher, seed, hashes))
        selected = [r for r in rows if r["teacher"] == teacher]
        s = dict(gate_mean_ba=float(np.mean([r["gate"]["balanced_accuracy"] for r in selected])),
                 gate_mean_equal_type_accuracy=float(np.mean([r["gate"]["equal_type_accuracy"] for r in selected])),
                 test_type_range=bounds([r["sufficient_test_types"] for r in selected]),
                 dominant_type_mass_range=bounds([r["max_fine_type_fraction"] for r in selected]),
                 cf_logical_occurrence_overlap_range=bounds([r["logical_test_occurrence_overlap"]["cf"] for r in selected]),
                 test_positive_range=bounds([r["test_labels"]["positive"] for r in selected]),
                 test_negative_range=bounds([r["test_labels"]["negative"] for r in selected]))
        for arm in ("native", "cf"):
            s[arm + "_mean_ba"] = float(np.mean([r["model_metrics"][arm]["balanced_accuracy"] for r in selected]))
            s[arm + "_mean_equal_type_accuracy"] = float(np.mean([r["model_metrics"][arm]["equal_type_accuracy"] for r in selected]))
            s[arm + "_training_labels"] = {k: bounds([r["train_color_labels"][arm][k] for r in selected])
                                            for k in ("positive", "negative")}
        s["phase_label_ranges"] = {phase: {arm: {k: bounds([r["train_labels_by_phase"][phase][arm][k] for r in selected])
                                                        for k in ("positive", "negative")}
                                            for arm in ("native", "cf")}
                                   for phase in selected[0]["train_labels_by_phase"]}
        s["strata_confusion_sums"] = {}
        for stratum in ("eligible_active", "eligible_silent", "ineligible"):
            available = [r["strata"][stratum] for r in selected if stratum in r["strata"]]
            if available:
                s["strata_confusion_sums"][stratum] = {
                    arm: {k: sum(r[arm][k] for r in available) for k in ("n", "tp", "tn", "fp", "fn")}
                    for arm in ("native", "cf", "gate")}
        summary[teacher] = s
    record = dict(analysis="post-test descriptive audit; no new primary endpoint or training",
                  logical_key=["phase", "eligible", "root participation", "queried option occupied"],
                  model_metric_provenance="Original frozen test JSON; no new neural inference",
                  truth_checks="Raw-array logical recomputation agrees with all base/auxiliary/test COMMIT labels",
                  fine_key_checks="Reconstructed training/test unions equal stored keys and obey original hash split",
                  script_sha256=digest(Path(__file__)), input_sha256=hashes, rows=rows, summary=summary)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(analysis=record["analysis"], paired_test_occurrences=sum(r["queries"] for r in rows),
                          summary=summary), indent=2))


if __name__ == "__main__":
    main()
