#!/usr/bin/env python3
"""One new-seed M6B/Pedro-style own-plus-counterfactual portability pilot."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from dcl.multicf.runner import ROOT,task
from dcl.multicf.data import acquire
from dcl.multicf.protocol import freeze
from dcl.study.storage import atomic_json,file_hash


if __name__=="__main__":
    config=json.loads(Path("configs/multiteacher-counterfactual.json").read_text())
    config.update(seeds=[95101],updates=1500,checkpoints=[500,1500])
    freeze(ROOT/"own-proposal-pilot/protocol.json",dict(config=config,method="owncf",seed=95101,
        budgets={t:30000 if t=="COLOR_TRIALS" else 3000 for t in config["teachers"]},script_sha256=file_hash(__file__),
        scope="Exploratory one-seed mechanism-portability pilot; not a five-seed certification or exact retraining of original unavailable Pedro weights. Native and stored cf labels plus every distinct newly queried own confirmation are charged."))
    jobs=[]
    for t in config["teachers"]:
        budget=30000 if t=="COLOR_TRIALS" else 3000
        folder=ROOT/"datasets"/t.lower()/"s95101"/f"b{budget}-coverage1-alternate0"
        acquire(config,t,95101,budget,folder,True,False)
        jobs.append((config,t,"normalized",95101,budget,"owncf",True,False,1500,[500,1500],"own-proposal-pilot"))
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(task,jobs))
    atomic_json(ROOT/"own-proposal-pilot/index.json",rows)
