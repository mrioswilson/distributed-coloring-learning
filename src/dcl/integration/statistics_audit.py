"""Matched-data reconciliation and independent known-law power controls.

No historical p-value is replaced. New comparisons have separately declared
families and explicit scope; unavailable Pedro checkpoints are never imputed.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from dcl.behavior_statistics import kernel_test, holm_adjust, write_csv
from dcl.globaldist.core import graph_record
from dcl.integration.study import OUT as STUDY_OUT
from dcl.multiteacher.programs import named_seed
from dcl.robustdist.controls import draw
from dcl.robustdist.stats import joint_test
from dcl.sizedegree.common import rng as original_factorial_rng
from dcl.study.storage import atomic_json, file_hash

OUT = Path("results/integration/statistics")


def cp(k, n, alpha=.05):
    return [0. if not k else float(beta.ppf(alpha/2, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-alpha/2, k+1, n-k))]


def seed_rng(*parts):
    return np.random.default_rng(named_seed("independent-integration-statistics-v1", *parts))


def pedro_test(x, y, B, *coordinates):
    r = kernel_test(x, y, np.zeros(len(x), np.int8), np.zeros(len(y), np.int8),
                    dict(kernel_scales=[1., 4., 16.], permutations=B), seed_rng(*coordinates))
    return {k: v for k, v in r.items() if k != "null"}


def matched_archives(root):
    """Both procedures on identical fixed historical graphs/checkpoints/banks."""
    OUT.mkdir(parents=True, exist_ok=True); root = Path(root)
    graphs = {g["id"]: g for g in json.loads((root / "results/ucgnn-size-degree/graphs.json").read_text())}
    historical = {(r["graph"], r["model"]): r for r in json.loads((root / "results/ucgnn-size-degree/main.json").read_text())}
    rows = []
    for gid in ("main-regular-n128-d3-r0", "main-regular-n512-d100-r0"):
        g = graphs[gid]
        teacher_path = root / "data/ucgnn-size-degree/banks" / f"confirmation__Luby__{gid}.npz"
        teacher = dict(np.load(teacher_path, allow_pickle=False))
        for label in ("normalized-s41000", "deepsets-s41000", "transformer-s41000"):
            model_path = root / "data/ucgnn-size-degree/banks" / f"confirmation__{label}__{gid}.npz"
            model = dict(np.load(model_path, allow_pickle=False)); old = historical[gid, label]
            x = teacher["colors"][:500][teacher["status"][:500] == 0]
            y = model["colors"][:500][model["status"][:500] == 0]
            assert len(x) == len(y) == 500
            v2 = joint_test(x, y, g, original_factorial_rng("confirmation", gid, label, "omnibus"), B=19999,
                            futility=20 if old["robust"]["p_kind"] == "conservative_futility" else None)
            assert v2["p"] == old["p"] and v2["statistics"] == old["robust"]["statistics"]
            mixture = pedro_test(x, y, 19999, "matched-archive", gid, label)
            rows.append(dict(graph=gid, model=label, samples=500, B=19999,
                teacher_bank=str(teacher_path), teacher_sha256=file_hash(teacher_path),
                model_bank=str(model_path), model_sha256=file_hash(model_path),
                v2_p=v2["p"], mixture_p=mixture["p_value"], mixture_mmd2=mixture["mmd2"],
                original_v2_holm=old["p_holm"], original_v2_reject=old["reject"],
                original_statistics_reproduced=True))
            print("matched archive", gid, label, flush=True)
    for key in ("v2_p", "mixture_p"):
        for r, adjusted in zip(rows, holm_adjust([r[key] for r in rows])):
            r[key + "_holm6"] = float(adjusted)
    atomic_json(OUT / "matched-archives.json", dict(comparisons=rows,
        scope="Exploratory analysis on already-observed historical banks. Six-comparison method-specific families; original 720-hypothesis Holm decisions remain unchanged. This uses our checkpoint cohort, not Pedro's missing weights."))
    write_csv(OUT / "matched-archives.csv", rows)


def independent_power():
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    for case in ("star_null", "localized"):
        for m in (256, 500):
            for trial in range(20):
                x, g, meta = draw(case, m, seed_rng("power-bank", case, m, trial, "A"), side=0)
                y, _, _ = draw(case, m, seed_rng("power-bank", case, m, trial, "B"), side=1)
                old = pedro_test(x, y, 999, "power-permutation-mixture", case, m, trial)
                new = joint_test(x, y, g, seed_rng("power-permutation-v2", case, m, trial), B=999)
                rows.append(dict(case=case, m=m, trial=trial, B=999, true_tv=meta["tv"],
                                 mixture_p=old["p_value"], v2_p=new["p"]))
            write_csv(OUT / "independent-power-trials.csv", rows)
            print("independent control", case, m, flush=True)
    summary = []
    for case, m, method in itertools.product(("star_null", "localized"), (256, 500), ("mixture", "v2")):
        group = [r for r in rows if r["case"] == case and r["m"] == m]
        count = sum(r[method + "_p"] <= .05 for r in group)
        summary.append(dict(case=case, m=m, method=method, detections=count, trials=len(group),
                            rate=count/len(group), pointwise_cp95=cp(count, len(group))))
    D = 200
    theoretical = []
    for gamma in (1., 4., 16.):
        a = np.exp(-gamma / (D+1)); b = (1+a)/2
        theoretical.append(float(b**D * (1-a) * (1/100-1/199)))
    atomic_json(OUT / "independent-power.json", dict(summary=summary, known_alternative_tv=99/199,
        hamming_mmd2_by_scale=theoretical, mixture_population_mmd2=float(np.mean(theoretical)),
        scope="Twenty independent experiments per law/size, B=999, nominal alpha=.05. CP intervals are pointwise, not simultaneous; no claim of strict 720-test-family power or universal calibration. Both methods use the same independently generated complete-coloring rows. No model or procedure was selected from these outcomes."))
    write_csv(OUT / "independent-power.csv", summary)


def new_study_statistics():
    """Unchanged V2 on each new teacher's corresponding reference law."""
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("configs/integration-balanced.json").read_text())
    graphs = {g["id"]: g for g in json.loads((STUDY_OUT / "graphs.json").read_text())}
    functional = json.loads((STUDY_OUT / "functional.json").read_text())
    references = {(r["teacher"], r["graph"]): r for r in functional if r["architecture"] == "exact"}
    rows = []
    for record in functional:
        if record["architecture"] == "exact":
            continue
        path = OUT / "new-study-tests" / f"{record['teacher']}__{record['graph']}__{record['label']}.json"
        if path.exists():
            rows.append(json.loads(path.read_text())); continue
        ref = references[record["teacher"], record["graph"]]
        assert file_hash(ref["bank"]) == ref["sha256"] and file_hash(record["bank"]) == record["sha256"]
        a, b = dict(np.load(ref["bank"], allow_pickle=False)), dict(np.load(record["bank"], allow_pickle=False))
        x, y = a["colors"][a["status"] == 0], b["colors"][b["status"] == 0]
        meta = graphs[record["graph"]]
        # Native archive edges are undirected pairs; the audited instrument
        # expects the same symmetric directed-edge serialization as earlier.
        edges = np.asarray(meta["edges"], int).reshape(-1, 2)
        directed = np.concatenate((edges, edges[:, ::-1]), axis=0).T.tolist()
        g = dict(id=record["graph"], n=meta["n"], degrees=meta["degrees"], edges=directed)
        result = dict(teacher=record["teacher"], architecture=record["architecture"], seed=record["seed"],
                      graph=record["graph"], valid_teacher=len(x), valid_model=len(y),
                      teacher_bank_sha256=ref["sha256"], model_bank_sha256=record["sha256"])
        if min(len(x), len(y)) < 4:
            result.update(testable=False, p=1., reason="Fewer than four valid completed rows; equality is not inferred.")
        else:
            stat = joint_test(x, y, g, seed_rng("new-study-v2", record["teacher"], record["graph"], record["label"]),
                              B=config["permutations"])
            result.update(testable=True, p=stat["p"], statistic=stat["statistics"]["omnibus"],
                          selection=stat["selection_per_class"], evaluation=stat["evaluation_per_class"])
        atomic_json(path, result); rows.append(result)
        if len(rows) % 25 == 0:
            print("new global tests", len(rows), flush=True)
    planned = len(config["teachers"]) * len(config["architectures"]) * len(config["seeds"]) * len(config["graphs_for_evaluation"])
    assert len(rows) == planned
    for r, adjusted in zip(rows, holm_adjust([r["p"] for r in rows])):
        r.update(p_holm=float(adjusted), reject=bool(adjusted <= .05))
    atomic_json(OUT / "new-study-global.json", dict(family_size=planned, B=config["permutations"],
        estimand="Labeled complete-coloring law conditional on proper native completion; each teacher has its own exact reference. p=1 for unavailable tests is bookkeeping, never equality evidence.", comparisons=rows))
    write_csv(OUT / "new-study-global.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("matched", "power", "new-study"))
    p.add_argument("--archive-root", default="/workspaces/AlgoLearning/ucgnn-size-degree-study")
    args = p.parse_args(); torch.set_num_threads(1)
    if args.stage == "matched": matched_archives(args.archive_root)
    elif args.stage == "power": independent_power()
    else: new_study_statistics()
