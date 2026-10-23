"""Prospective exploratory controls. None consume final confirmation data."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from dcl.study.storage import atomic_json, file_hash
from .data import acquire
from .training import initialize, train


def merge(passive, interactive, destination):
    """Cache aggregation counts all acquired labels; no additional oracle call."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    first,second=[np.load(Path(p)/"dataset.npz",allow_pickle=False) for p in (passive,interactive)]
    ms=[json.loads((Path(p)/"manifest.json").read_text()) for p in (passive,interactive)]
    arrays={k:np.concatenate((first[k],second[k]+ms[0]["independent_trajectories"] if k=="trajectory" else second[k])) for k in first.files}
    np.savez_compressed(destination/"dataset.npz",**arrays)
    counts=Counter(ms[0]["by_component"]);counts.update(ms[1]["by_component"])
    trajectories=ms[0]["trajectories"]+[dict(r,trajectory=r["trajectory"]+ms[0]["independent_trajectories"]) for r in ms[1]["trajectories"]]
    manifest=dict(identity=dict(mode="one-aggregation-DAgger-like-control"),queries=sum(m["queries"] for m in ms),
                  by_component=dict(counts),independent_trajectories=sum(m["independent_trajectories"] for m in ms),
                  trajectories=trajectories,graphs={**ms[0]["graphs"],**ms[1]["graphs"]},
                  sha256=file_hash(destination/"dataset.npz"),unique_stored_bytes=(destination/"dataset.npz").stat().st_size,
                  sources=[dict(path=str(p),sha256=file_hash(Path(p)/"manifest.json")) for p in (passive,interactive)])
    atomic_json(destination/"manifest.json",manifest)
    return manifest


def worker(job):
    config,root,teacher,seed=job
    root=Path(root);budget=config["control_budget"];records=[]
    dataset=root/"datasets"/teacher.lower()/f"s{seed}"
    # Small and no-message controls use exactly the same cached budget prefix.
    for arch in config["control_architectures"]:
        records.append(dict(kind="capacity_or_communication",record=train(config,teacher,arch,seed,budget,dataset,
                       root/"controls"/"capacity"/teacher.lower()/arch/f"s{seed}")))
    # More updates with identical labels and identical initialization.
    records.append(dict(kind="fixed_labels_more_optimization",record=train(config,teacher,"normalized",seed,budget,dataset,
                   root/"controls"/"optimization"/teacher.lower()/f"s{seed}",updates=max(config["optimization_control_updates"]))))
    # One declared DAgger-like aggregation, admissible states only. Same total
    # 500 updates as passive, with a new optimizer at the aggregation boundary.
    passive_config=dict(config,namespace=config["namespace"]+"-interactive-passive")
    on_config=dict(config,namespace=config["namespace"]+"-interactive-on-policy")
    folder=root/"controls"/"interactive"/teacher.lower()/f"s{seed}"
    acquire(passive_config,teacher,seed,budget//2,folder/"passive")
    warm=train(config,teacher,"normalized",seed,budget//2,folder/"passive",folder/"warmup",
               updates=config["updates"]//2,dev=False)
    model=initialize(teacher,"normalized",seed)
    model.load_state_dict(torch.load(folder/"warmup"/"final.pt",weights_only=False)["model"]);model.eval()
    on=acquire(on_config,teacher,seed,budget-budget//2,folder/"on-policy",policy=model)
    merged=merge(folder/"passive",folder/"on-policy",folder/"merged")
    record=train(config,teacher,"normalized",seed,budget,folder/"merged",folder/"final",
                 updates=config["updates"]-config["updates"]//2,warmstart=folder/"warmup"/"final.pt")
    record["interactive"]=dict(warmup_updates=config["updates"]//2,total_updates=config["updates"],
         total_queries=merged["queries"],on_policy_queries=on["queries"],
         failed_on_policy_trajectories=sum(r.get("failure") is not None for r in on["trajectories"]),
         warmup_seconds=warm["training_seconds"],
         caveat="One aggregation, frozen warm policy during acquisition, Adam reset at aggregation. This is not a general DAgger efficiency theorem. GM excluded for unproved history admissibility.")
    records.append(dict(kind="interactive",record=record))
    atomic_json(Path("results/sample-complexity/controls")/teacher.lower()/f"s{seed}.json",records)
    return records


def run(config,root):
    from .runner import verify
    verify(config)
    jobs=[(config,str(root),t,s) for s in config["seeds"] for t in config["optimization_control_teachers"]]
    with ProcessPoolExecutor(config["workers"],mp_context=multiprocessing.get_context("spawn")) as executor:
        records=list(executor.map(worker,jobs))
    atomic_json("results/sample-complexity/controls.json",[r for group in records for r in group])
