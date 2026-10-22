import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,time
import numpy as np
import torch
from .core import *
from .sampling import sample
from dcl.localpalettes.runner import configure,load,persistent_gpu_lock

def main():
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    configure();rows=[];t=time.monotonic()
    selected=[small_graphs()[1]]+[g for g in graphs() if g['id'] in ('regular-n256-d20','regular-n512-d200','star-d200','random-n256-d100')]
    with persistent_gpu_lock():
        for spec in [s for s in specs() if s['seed']==41000]:
            model=load(spec)
            for g in selected:
                a=sample(g,model,spec['label'],'pilot-cost',12,batch_nodes=1024)
                np.savez_compressed(ROOT/f"pilot-{spec['label']}-{g['id']}.npz",**a)
                rows.append(dict(model=spec['label'],graph=g['id'],n=g['n'],delta=g['delta'],m=12,seconds=float(a['seconds']),
                                 statuses=np.bincount(a['status'],minlength=4).tolist()))
                print(json.dumps(rows[-1]),flush=True)
            # Same fresh run replay, interleaved with a different state; adapter recreated.
            a=sample(selected[0],model,spec['label'],'pilot-reset',12,batch_nodes=48)
            sample(selected[-1],model,spec['label'],'pilot-interleaved',1)
            b=sample(selected[0],model,spec['label'],'pilot-reset',12,batch_nodes=48)
            assert np.array_equal(a['colors'],b['colors'])
            # Compare batching across disjoint copies on actual frozen policy.
            c=sample(selected[0],model,spec['label'],'pilot-reset',12,batch_nodes=12)
            assert np.array_equal(a['colors'],c['colors'])
            del model;torch.cuda.empty_cache()
    # Independent pilot sensitivity, not used in confirmatory tests.
    g=small_graphs()[1];x,p,_,_=exact_teacher(g)
    controls=[]
    for i in range(100):
        rng=np.random.default_rng(seed('pilot-controls',i));a=x[rng.choice(len(x),100,p=p)];b=x[rng.choice(len(x),100,p=p)]
        null=mmd_test(a,b,rng,B=199)
        alt=mmd_test(np.repeat(x[:1],100,axis=0),np.repeat(x[-1:],100,axis=0),rng,B=199)
        controls.append(dict(null_p=null['p'],positive_p=alt['p']))
    atomic_json(OUT/'pilot.json',dict(rows=rows,controls=controls,gpu_seconds=time.monotonic()-t,
        null_rejections=sum(r['null_p']<=.05 for r in controls),positive_rejections=sum(r['positive_p']<=.05 for r in controls)))
if __name__=='__main__':main()
