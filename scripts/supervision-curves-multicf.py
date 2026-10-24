#!/usr/bin/env python3
"""Nested finite counterfactual banks at fixed optimizer budgets (development)."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from dcl.study.storage import atomic_json,file_hash
from dcl.multicf.data import load
from dcl.multicf.training import train

OUT=Path("results/multiteacher-counterfactual/supervision-curves")
MAIN=Path("results/multiteacher-counterfactual")


def prefix(seed,budget):
    source=MAIN/"datasets/color_trials"/f"s{seed}"/"b30000-coverage1-alternate0"
    packed,manifest=load(source);take=budget//3
    directory=OUT/"datasets"/f"s{seed}"/f"b{budget}";directory.mkdir(parents=True,exist_ok=True)
    target=directory/"dataset.npz"
    if target.exists():return directory
    np.savez_compressed(target,**{k:v[:take] for k,v in packed.items()})
    counts={c:take for c in ("participation","proposal","confirmation")}
    identity=dict(manifest["identity"],native_budget=budget)
    atomic_json(directory/"manifest.json",dict(identity=identity,native_labels=take*3,counterfactual_labels=take*4,
        total_acquired_component_labels=take*7,by_native_component=counts,examples=take,
        source_sha256=manifest["sha256"],sha256=file_hash(target),source_directory=str(source),
        scope="Exact nested root and counterfactual-query prefix of the common 30000-native-label bank; no new teacher labels acquired to construct a prefix"))
    return directory


def job(args):
    config,arch,seed,budget=args
    directory=OUT/arch/f"s{seed}"/f"b{budget}"
    return train(config,"COLOR_TRIALS",arch,seed,OUT/"datasets"/f"s{seed}"/f"b{budget}",directory,
                 "counterfactual",1500,[500,1500])


if __name__=="__main__":
    config=json.loads(Path("configs/multiteacher-counterfactual.json").read_text());torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/"protocol.json",dict(config=config,budgets=[1000,3000,10000,30000],updates=[500,1500],
        source_of_30000="Reuse primary checkpoints; do not retrain",script_sha256=file_hash(__file__),
        scope="Exploratory development curves after initial method design; no new final PASS or minimum-budget claim. Same initialization and fixed acquired prefixes across optimization points."))
    for seed in config["seeds"]:
        for budget in (1000,3000,10000):prefix(seed,budget)
    with ProcessPoolExecutor(max_workers=3) as pool:
        result=list(pool.map(job,[(config,a,s,b) for a in config["architectures"] for s in config["seeds"] for b in (1000,3000,10000)]))
    atomic_json(OUT/"index.json",result)
