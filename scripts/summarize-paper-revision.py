#!/usr/bin/env python3
"""Paired-seed summaries; no model selection, query-level intervals or training."""
import argparse
import json
from pathlib import Path
import numpy as np

OUT=Path("results/paper-revision-20261030")


def interval(differences,rng):
    d=np.asarray(differences,float)
    samples=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
    return [float(x) for x in np.quantile(samples,[.025,.975])]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--runtime",type=Path,default=Path("data/paper-revision-20261030"));args=parser.parse_args()
    config=json.loads(Path("configs/paper-revision-20261030.json").read_text())
    freeze=json.loads((OUT/"weights-frozen.json").read_text())
    assert len(freeze["weights"])==40
    rng=np.random.default_rng(2026103017);rows=[];summary=[];costs=[]
    for teacher in config["teachers"]:
        part=[]
        for seed in config["seeds"]:
            models={}
            for arm in ("native","cf"):
                path=OUT/"prospective"/teacher.lower()/f"s{seed}"/(arm+"-test.json")
                r=json.loads(path.read_text());models[arm]=r
                diagnostic=r["diagnostic"]
                assert diagnostic["train_test_type_intersection"]==0 and diagnostic["n"]==4000
                row=dict(teacher=teacher,seed=seed,arm=arm,**diagnostic,
                         successes=r["rollout_summary"]["successes"],runs=r["rollout_summary"]["executions"],
                         failures=r["rollout_summary"]["failures"],mean_steps=r["rollout_summary"]["mean_native_steps"])
                rows.append(row);part.append(row)
            manifest=json.loads((args.runtime/"datasets"/teacher.lower()/f"s{seed}"/"manifest.json").read_text())
            basepath=args.runtime/"datasets"/teacher.lower()/f"s{seed}"/"base.npz"
            with np.load(basepath,allow_pickle=False) as stored:
                color=~np.isin(stored["native"][:,3],[2,4])  # DETECT=2, LINIAL=4
                root_context_values=int(2*color.sum())
                neighbor_context_values=int(2*stored["a3"][color,0].sum())
            costs.append(dict(manifest,base_conditioning=dict(
                root_values=root_context_values,neighbor_values=neighbor_context_values,
                note="Base root P/C values overlap native target labels; neighbor values are shared conditioning occurrences. Counts are scalar occurrences, not distinct simulator calls."),
                shared_physical_optimization_targets_per_pair=10200))
        native=[r for r in part if r["arm"]=="native"];cf=[r for r in part if r["arm"]=="cf"]
        defined=all(r["balanced_accuracy"] is not None for r in part)
        differences=[c["balanced_accuracy"]-n["balanced_accuracy"] for n,c in zip(native,cf)] if defined else None
        summary.append(dict(teacher=teacher,seeds=10,
            native_mean_ba=float(np.mean([r["balanced_accuracy"] for r in native])) if defined else None,
            cf_mean_ba=float(np.mean([r["balanced_accuracy"] for r in cf])) if defined else None,
            paired_ba_differences=differences,mean_paired_ba_difference=float(np.mean(differences)) if defined else None,
            paired_seed_bootstrap_95=interval(differences,rng) if defined else None,
            native_mean_equal_type_accuracy=float(np.mean([r["equal_type_accuracy"] for r in native])),
            cf_mean_equal_type_accuracy=float(np.mean([r["equal_type_accuracy"] for r in cf])),
            native_successes=sum(r["successes"] for r in native),cf_successes=sum(r["successes"] for r in cf),
            runs_per_arm=sum(r["runs"] for r in native),
            native_mean_steps=float(np.mean([r["mean_steps"] for r in native])),cf_mean_steps=float(np.mean([r["mean_steps"] for r in cf])),
            success_differences=[c["successes"]-n["successes"] for n,c in zip(native,cf)],
            positive_queries_by_seed=[r["tp"]+r["fn"] for r in native]))
    payload=dict(rows=rows,summary=summary,costs=costs,weights_frozen_utc=freeze["created_utc"],
                 independent_training_replicates_per_teacher=10,
                 primary_interval="Percentile bootstrap of ten paired seed differences, 10000 resamples. Conditional on the fixed type partition, acquisition rule and sampled graph/tape domains; not a query-level confidence statement.")
    (OUT/"prospective-summary.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
