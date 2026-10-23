"""Pilot inference resource check, without reading any confirmatory outcome."""
import json
import time

import torch

from dcl.extreme.graphs import generate
from dcl.m4_data import _edge_index
from dcl.study.storage import training_lock,atomic_json,now,file_hash
from .config import ROOT,named_seed
from .models import PRIMARY
from .training import load_model,weights_hash
from .report import csv
from .mechanisms import dense_flops


@torch.inference_mode()
def main():
    path=ROOT/"manifest.json"
    manifest=json.loads(path.read_text())
    if manifest["status"]!="stopped_recoverable":
        raise RuntimeError("Run this one-off resource preflight at a quiescent coordinator boundary")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.5)
    rows=[]
    started=time.monotonic()
    with training_lock(ROOT):
        for treatment in PRIMARY:
            checkpoint=ROOT/"runs"/f"{treatment}-s41000"/"epoch-120.pt"
            model,_=load_model(checkpoint,"cuda")
            initial=weights_hash(model.state_dict())
            for n,d in ((160,3),(640,200)):
                edges=torch.tensor(_edge_index(generate(n,d,"REGULAR",named_seed("preflight",n,d))),device="cuda")
                colors=torch.full((n,),-1,dtype=torch.long,device="cuda")
                rp=torch.full((n,),.25,device="cuda")
                rc=torch.full((n,),.63,device="cuda")
                palette=torch.full((n,),d+1,dtype=torch.long,device="cuda")
                tick=time.monotonic()
                try:
                    model(colors,rp,rc,edges,palette,8192)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    tick=time.monotonic()
                    for _ in range(3):model(colors,rp,rc,edges,palette,8192)
                    torch.cuda.synchronize()
                    result={"status":"ok","forward_seconds":(time.monotonic()-tick)/3,"gpu_peak_bytes":torch.cuda.max_memory_allocated()}
                except torch.OutOfMemoryError as e:
                    result={"status":"oom","error":str(e),"wall_seconds":time.monotonic()-tick}
                    torch.cuda.empty_cache()
                rows.append({"treatment":treatment,"n":n,"degree":d,"palette":d+1,"checkpoint_sha256":file_hash(checkpoint),
                             **dense_flops(model,n,d+1,edges.shape[1]),**result})
                csv("results/ucgnn-paper-study/resources-preflight.csv",rows)
                print({k:v for k,v in rows[-1].items() if k in ("treatment","n","degree","status","forward_seconds","gpu_peak_bytes")},flush=True)
            if weights_hash(model.state_dict())!=initial:raise RuntimeError("Resource check altered weights")
            del model
            torch.cuda.empty_cache()
    elapsed=time.monotonic()-started
    manifest=json.loads(path.read_text())
    if manifest["status"]!="stopped_recoverable":raise RuntimeError("Coordinator resumed during preflight")
    manifest["active_seconds"]+=elapsed
    manifest.setdefault("manual_budget_events",[]).append({"action":"pilot_resource_preflight","seconds":elapsed,"utc":now(),
                                                          "outcomes_examined":False,"confirmatory_graphs_used":False})
    atomic_json(path,manifest)
    print("Accounted active GPU process seconds:",elapsed,flush=True)


if __name__=="__main__":
    main()
