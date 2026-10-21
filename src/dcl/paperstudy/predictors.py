"""Frozen weight-derived palette-quantile predictions (prospective addendum A1)."""
import json
import numpy as np

from dcl.study.storage import atomic_json,file_hash,now


def limiting_quantile(coefficients,grid):
    b,a,r,v=[np.asarray(coefficients[k],dtype=np.float64) for k in ("B","A","R","v")]
    output=[]
    for start in range(0,len(grid),128):
        draws=np.asarray(grid[start:start+128],np.float64)
        offset=b[None,:]+draws[:,None]*r
        roots=np.divide(-offset,a[None,:],out=np.zeros_like(offset),where=a[None,:]!=0)
        xs=np.sort(np.c_[np.zeros(len(draws)),np.ones(len(draws)),np.clip(roots,0,1)],axis=1)
        scores=np.maximum(offset[:,None,:]+xs[:,:,None]*a,0)@v
        selected=np.argmax(scores,axis=1)
        output.append(xs[np.arange(len(draws)),selected])
    return np.concatenate(output)


def freeze_analytic(model,checkpoint,path):
    from .mechanisms import center_state
    checkpoint_hash=file_hash(checkpoint)
    if path.exists():
        saved=json.loads(path.read_text())
        if saved["checkpoint_sha256"]!=checkpoint_hash:raise RuntimeError("Analytic predictor checkpoint mismatch")
        return saved
    matrix=model.scorer[0].weight.detach().cpu().double().numpy()
    bias=model.scorer[0].bias.detach().cpu().double().numpy()
    v=model.scorer[2].weight.detach().cpu().double().numpy().reshape(-1)
    contexts={}
    for key,free in (("u1f0",101),("u1f1",51),("u0f1",1),("u0f0",101)):
        colors=np.full(1 if key=="u0f0" else 101,-1,np.int64)
        if free<101:colors[1:1+101-free]=np.arange(free,101)
        state=center_state(model,colors,101)
        h=state[0][0].detach().cpu().double().numpy()
        alpha=float(state[1][0])
        b=matrix[:,:32]@h+matrix[:,32]*alpha+bias
        contexts[key]={"calibration_palette":101,"calibration_available":free,"calibration_neighbors":len(colors)-1,
                       "h":h.tolist(),"alpha":alpha,"B":b.tolist(),"A":(matrix[:,33]+matrix[:,34]).tolist(),
                       "R":matrix[:,35].tolist(),"C":matrix[:,34].tolist(),"v":v.tolist()}
    saved={"checkpoint":str(checkpoint),"checkpoint_sha256":checkpoint_hash,"contexts":contexts,
           "global_context":"u1f0","device":str(next(model.parameters()).device),"dtype":str(next(model.parameters()).dtype),
           "numeric_prediction":"float64 continuous ReLU maximizer at width zero, smallest maximizer, floor(k*x)",
           "heldout_not_evaluated_yet":True,"frozen_utc":now()}
    atomic_json(path,saved)
    return saved


def predict_ranks(frozen,grid,count,colors):
    context=f"u{int(np.any(np.asarray(colors)[1:]<0))}f{int(np.any(np.asarray(colors)[1:]>=0))}"
    result={}
    for label,key in (("analytic_global",frozen["global_context"]),("analytic_context",context)):
        quantile=limiting_quantile(frozen["contexts"][key],grid)
        result[label]=np.floor(count*quantile).astype(int).clip(0,count-1)
    return result
