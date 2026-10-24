#!/usr/bin/env python3
"""Paired 4000-update diagnostics, independent of candidate selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

import torch

from dcl.study.storage import atomic_json

spec=importlib.util.spec_from_file_location("multicf_analysis",Path(__file__).with_name("analyze-multicf.py"))
analysis=importlib.util.module_from_spec(spec);sys.modules[spec.name]=analysis;spec.loader.exec_module(analysis)
OUT=Path("results/multiteacher-counterfactual")


def job(args):
    config,t,a,s,m,b,u=args
    path=OUT/"runs"/t.lower()/a/f"s{s}"/f"b{b}-coverage1-alternate0"/m/f"u{u:06d}.pt"
    result=analysis.diagnostics(config,t,a,s,path,f"paired-u{u:06d}-{m}")
    print(json.dumps(dict(diagnostic=t,architecture=a,seed=s,method=m,action=result["action_fidelity"])),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--teachers",nargs="+",default=["BE_BASE_D3","SPECULATIVE_GM","COLOR_TRIALS"])
    p.add_argument("--methods",nargs="+",default=["passive","matched","counterfactual"])
    p.add_argument("--updates",type=int,default=4000);p.add_argument("--workers",type=int,default=3);a=p.parse_args()
    config=json.loads(Path("configs/multiteacher-counterfactual.json").read_text());torch.set_num_threads(1)
    for t in a.teachers:
        for s in config["seeds"]:analysis.evaluation_bank(config,t,s)
    jobs=[(config,t,arch,s,m,30000 if t=="COLOR_TRIALS" else 3000,a.updates)
          for t in a.teachers for arch in config["architectures"] for s in config["seeds"] for m in a.methods]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(job,jobs))
    atomic_json(OUT/"diagnostics"/f"paired-u{a.updates:06d}-{'-'.join(a.teachers)}.json",results)
