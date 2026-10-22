from __future__ import annotations
import time,json,hashlib
import numpy as np
import torch
from dcl.localpalettes.adapter import LocalPaletteAdapter,converted_uniform
from dcl.universal.rules import teacher_offer,exact_commit_labels
from .core import seed


def sample(graph,model,label,stage,m,rounds=128,batch_nodes=8192,offset=0):
    """Independent PCG64 stream per complete run; batching is disjoint-union execution."""
    n=graph['n'];e=np.asarray(graph['edges'],dtype=np.int64).reshape(2,-1);d=np.asarray(graph['degrees']);pal=d+1
    seeds=[seed(stage,graph['id'],label,'run',i) for i in range(offset,offset+m)]
    colors=np.full((m,n),-1,dtype=np.int16);status=np.full(m,3,dtype=np.int8);nr=np.zeros(m,dtype=np.int16)
    errors=[];final_states=[];start=time.monotonic()
    for lo in range(0,m,max(1,batch_nodes//n)):
        hi=min(m,lo+max(1,batch_nodes//n));b=hi-lo;rngs=[np.random.default_rng(s) for s in seeds[lo:hi]]
        c=np.full((b,n),-1,dtype=np.int64)
        ee=np.concatenate([e+j*n for j in range(b)],axis=1);pp=np.tile(pal,b)
        adapter=LocalPaletteAdapter(model) if model is not None else None
        if model is not None:
            device=next(model.parameters()).device;et=torch.as_tensor(ee,device=device);pt=torch.as_tensor(pp,device=device)
        try:
            for r in range(rounds):
                active=(c<0).any(1)
                if not active.any():break
                nr[lo:hi]+=active
                random=np.stack([rng.random((n,2)) for rng in rngs]);rp=random[:,:,0].reshape(-1);rc=random[:,:,1].reshape(-1)
                flat=c.reshape(-1)
                if model is None:
                    cand,_=teacher_offer(flat,rc,ee,pp);part=(rp<.5)&(flat<0);commit=exact_commit_labels(flat,ee,part,cand)
                else:
                    with torch.inference_mode():
                        out=adapter(torch.as_tensor(flat,device=device),torch.as_tensor(converted_uniform(rp),device=device),
                                    torch.as_tensor(converted_uniform(rc),device=device),et,pt)
                    cand=out.candidate.cpu().numpy();part=out.participate.cpu().numpy()
                    commit=(out.commit_logit>0).cpu().numpy()&part&(flat<0)
                flat[commit]=cand[commit]
            colors[lo:hi]=c
            for j,x in enumerate(c):
                # 0 valid complete; 1 invalid complete; 2 cutoff; 3 technical error.
                status[lo+j]=2 if (x<0).any() else (0 if (x<pal).all() and (x>=0).all() and (x[e[0]]!=x[e[1]]).all() else 1)
        except Exception as exc:
            colors[lo:hi]=c;errors.append(dict(start=lo,end=hi,error=repr(exc)))
        final_states.extend([rng.bit_generator.state for rng in rngs])
    return dict(first_rng_initial_state=np.array(json.dumps(np.random.default_rng(seeds[0]).bit_generator.state)),
                final_rng_states_sha256=np.array(hashlib.sha256(json.dumps(final_states,sort_keys=True).encode()).hexdigest()),
                colors=colors,status=status,rounds=nr,seeds=np.asarray(seeds,dtype=np.uint64),seconds=np.array(time.monotonic()-start),errors=np.array(str(errors)))
