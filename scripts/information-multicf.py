#!/usr/bin/env python3
"""JP information ablation: same parameter tensors, labels, batches and updates."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from dcl.integration.models import RootedPalettePolicy
from dcl.multiteacher.programs import named_seed
from dcl.multiteacher.experiment import weight_hash
from dcl.samplecomplexity.model import QueryPolicy
from dcl.samplecomplexity.training import initialize
from dcl.samplecomplexity.evaluation import evaluate
from dcl.study.storage import atomic_json,save_checkpoint,file_hash
from dcl.multicf.data import acquire,load
from dcl.multicf.protocol import cases,summarize_runs
from dcl.multicf.training import objective

OUT=Path("results/multiteacher-counterfactual/information")
MODES=("without_ids","numeric_ids","numeric_differences","relative_comparisons")


class InformationPolicy(QueryPolicy):
    def __init__(self,mode):
        super().__init__("JP","normalized");self.mode=mode
        if mode=="numeric_ids":
            self.context_message1.register_forward_pre_hook(self.raw_ids1)
            self.context_message2.register_forward_pre_hook(self.raw_ids2)

    def raw_ids1(self,module,args):
        z=args[0].clone();x,y,_,_=self.original_arrays
        z[...,6]=x[...,None,5].expand_as(z[...,6]);z[...,7]=y[...,5]
        return (z,)

    def raw_ids2(self,module,args):
        z=args[0].clone();x,_,_,_=self.original_arrays
        z[...,6]=x[:,:1,5,None].expand_as(z[...,6]);z[...,7]=x[:,1:,5,None].expand_as(z[...,7])
        return (z,)

    def first(self,arrays,phase):
        self.original_arrays=arrays
        if self.mode=="relative_comparisons":return super().first(arrays,phase)
        if self.mode=="without_ids":
            x,y,r,m=arrays;x,y=x.clone(),y.clone();x[...,5]=0;y[...,5]=0;arrays=(x,y,r,m)
        return RootedPalettePolicy.first(self,arrays,phase)


def job(args):
    config,seed,mode=args;torch.set_num_threads(1)
    directory=OUT/mode/f"s{seed}";directory.mkdir(parents=True,exist_ok=True)
    result_path=directory/"record.json"
    if result_path.exists():return json.loads(result_path.read_text())
    dataset=OUT/"datasets"/f"s{seed}"
    packed,manifest=load(dataset)
    base=initialize("JP","normalized",seed)
    model=InformationPolicy(mode);model.load_state_dict(base.state_dict());initial=weight_hash(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    order=np.random.default_rng(named_seed(config["namespace"],"information-batches",seed))
    tick,cpu=time.perf_counter(),time.process_time()
    for u in range(500):
        positions=order.choice(len(packed["native"]),64,replace=True)
        optimizer.zero_grad(set_to_none=True)
        loss,_,_=objective(model,packed,positions,"matched",u,100,0)
        loss.backward();optimizer.step()
    wall,proc=time.perf_counter()-tick,time.process_time()-cpu
    cp=directory/"final.pt";save_checkpoint(cp,dict(model=model.state_dict(),mode=mode,seed=seed,optimizer=optimizer.state_dict()))
    suite=cases(config,"development",seed);rows,seconds=evaluate(model,"JP",suite)
    atomic_json(directory/"development.json",rows)
    record=dict(teacher="JP",architecture="normalized",mode=mode,seed=seed,updates=500,batch=64,lr=.001,
        initial_hash=initial,weights_hash=weight_hash(model),data_sha256=manifest["sha256"],native_labels=manifest["native_labels"],
        parameters=sum(p.numel() for p in model.parameters()),training_seconds=wall,cpu_seconds=proc,gpu_seconds=0.,
        checkpoint_sha256=file_hash(cp),script_sha256=file_hash(__file__),development=summarize_runs(rows,[m["case_id"] for _,m in suite]),
        development_seconds=seconds,scope="Exploratory development only; no final certification. Identical dimensions and initialization; two JP context channels encode raw endpoint IDs in numeric_ids, edge difference plus zero in numeric_differences, relative order plus zero in relative_comparisons, zeros in without_ids.")
    atomic_json(result_path,record);print(json.dumps(dict(information=mode,seed=seed,development=record["development"]["successes"])),flush=True)
    return record


if __name__=="__main__":
    config=json.loads(Path("configs/multiteacher-counterfactual.json").read_text())
    config.update(namespace=config["namespace"]+"-information",development_runs=60)
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/"protocol.json",dict(config=config,modes=MODES,updates=500,budget=3000,
        selection="All four modes and all five seeds; no tuning or dropping a mode",script_sha256=file_hash(__file__)))
    for seed in config["seeds"]:acquire(config,"JP",seed,3000,OUT/"datasets"/f"s{seed}",coverage=True)
    with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(job,[(config,s,m) for s in config["seeds"] for m in MODES]))
    atomic_json(OUT/"results.json",results)
