#!/usr/bin/env python3
"""Independent paired optimizer follow-up for the retained ColorTrials failure."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

import torch

from dcl.study.storage import atomic_json,file_hash
from dcl.multicf.protocol import freeze,source_hashes

spec=importlib.util.spec_from_file_location("multicf_confirmation",Path(__file__).with_name("confirm-multicf.py"))
confirmation=importlib.util.module_from_spec(spec);sys.modules[spec.name]=confirmation;spec.loader.exec_module(confirmation)
OUT=Path("results/multiteacher-counterfactual")
PLAN=OUT/"plans/ct-cf-optimization-v2.json"


def execute(args):return confirmation.confirm(args)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["freeze","run"]);p.add_argument("--workers",type=int,default=3);a=p.parse_args()
    torch.set_num_threads(1)
    if a.stage=="freeze":
        config=json.loads(Path("configs/multiteacher-counterfactual.json").read_text())
        config["namespace"]+="-ct-cf-optimization-v2"
        selections=[]
        for arch in config["architectures"]:
            for u in (1500,4000):
                cps=[]
                for s in config["seeds"]:
                    path=OUT/"runs/color_trials"/arch/f"s{s}"/"b30000-coverage1-alternate0/counterfactual"/f"u{u:06d}.pt"
                    cps.append(dict(seed=s,path=str(path),sha256=file_hash(path)))
                selections.append(dict(teacher="COLOR_TRIALS",architecture=arch,budget=30000,method="counterfactual",
                    update=u,name=f"ct-cf-optimization-v2-{arch}-u{u}",checkpoints=cps,
                    selection="Both existing fixed-label optimizer points, all cores and five seeds; frozen before this fresh final battery"))
        freeze(PLAN,dict(config=config,selections=selections,sources=source_hashes(),script_sha256=file_hash(__file__),
            reason="Exploratory follow-up after original ColorTrials/E1 counterfactual u1500 seed 94103 had 999/1000. No weights are refitted and the old failure remains. Both optimizer points are reported for every architecture under a new namespace.",
            interpretation="Independent validation of fixed points; no minimum-label, universal-correctness or retroactive primary PASS claim"))
        print({"frozen_packages":len(selections),"new_namespace":config["namespace"]})
    else:
        plan=json.loads(PLAN.read_text())
        if plan["script_sha256"]!=file_hash(__file__) or plan["sources"]!=source_hashes():raise RuntimeError("Follow-up source changed")
        with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(execute,[(plan["config"],s) for s in plan["selections"]]))
        atomic_json(OUT/"confirmation-ct-cf-optimization-v2.json",results)
