"""Explicit CPU cross-check/pilot, and queued CUDA equivalence gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import atomic_json, now, provenance, weights_hash
from .plan import tasks
from .runtime import ChunkedPolicy
from .worker import evaluate
from dcl.universal.train import load_model


@torch.inference_mode()
def equivalence(checkpoint, device, output):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model,saved=load_model(checkpoint,device)
    before=weights_hash(model)
    rng=np.random.default_rng(173031)
    rows=[]
    for degree in (3,10,20,50,100,200,500):
        n=40
        colors=rng.integers(-1,degree+1,n,dtype=np.int64)
        colors[::2]=-1
        edges=np.array([(i,(i+1)%n) for i in range(n)]+[((i+1)%n,i) for i in range(n)]).T
        args=(torch.tensor(colors,device=device),torch.tensor(rng.random(n),dtype=torch.float32,device=device),
              torch.tensor(rng.random(n),dtype=torch.float32,device=device),torch.tensor(edges,device=device),
              torch.full((n,),degree+1,dtype=torch.long,device=device))
        a=model(*args,edge_chunk=23)
        b=ChunkedPolicy(model,8)(*args)
        differences={key:float((getattr(a,key)-getattr(b,key)).abs().max()) for key in
                     ("candidate_logits","commit_logit","availability","k_hat","lo","hi")}
        for key in differences:
            torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=2e-5,atol=2e-5)
        assert torch.equal(a.candidate,b.candidate) and torch.equal(a.participate,b.participate)
        assert torch.equal(a.commit_logit>0,b.commit_logit>0)
        rows.append({"degree":degree,"max_absolute_differences":differences,"hard_decisions_identical":True})
    assert weights_hash(model)==before
    record={"passed":True,"device":device,"checkpoint":str(checkpoint),"epoch":saved["epoch"],
            "weights_sha256_before":before,"weights_sha256_after":weights_hash(model),"rows":rows,
            "provenance":provenance(),"utc":now()}
    atomic_json(output,record)
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--pilot",action="store_true")
    args=p.parse_args()
    root=Path(args.root).resolve()
    config=json.loads(Path(args.config).read_text())
    inventory=json.loads((root/"audit/inventory.json").read_text())
    model=next(r for r in inventory["runs"] if r["config"]["name"]=="E1" and r["config"]["seed"]==0)
    equivalence(model["checkpoint"],args.device,root/f"audit/equivalence-{args.device}.json")
    print(f"equivalence {args.device} passed",flush=True)
    if args.pilot:
        # A separate explicit CPU cohort; GPU final cohorts use their own seeds.
        config={**config,"namespace":config["namespace"]+1}
        pilot=root/f"{args.device}-pilot"
        for task in tasks(config,inventory):
            if task["stage"]!="A-functional": break
            path=pilot/f"{task['task_id']}.json"
            if path.exists(): continue
            result=evaluate(task,config,root,args.device)
            atomic_json(path,result)
            print(f"{args.device} pilot d{task['graph']['degree']} {task['graph']['family']} "
                  f"{result['status']} rounds={result.get('rounds')} "
                  f"unsafe={result.get('unsafe_commits')} seconds={result.get('total_seconds')}",flush=True)


if __name__=="__main__": main()
