"""Fresh unknown-location alternatives for validation round two."""
import numpy as np
from .controls import draw as draw_v1,NULLS,ALTERNATIVES as ALTERNATIVES_V1
ALTERNATIVES=ALTERNATIVES_V1+('correlation_remote','higher_order_remote')
def draw(case,m,r,side=0,D=200,effect=None):
    remote=case.endswith('_remote');base=case.removesuffix('_remote')
    z,g,meta=draw_v1(base,m,r,side,D,effect)
    if remote:
        ids=(137,181) if base=='correlation' else (97,155,197)
        order=np.arange(D+1)
        for a,b in zip(range(1,len(ids)+1),ids):order[a],order[b]=order[b],order[a]
        z=z[:,order];meta['unknown_interaction_region']=list(ids)
        meta['kind']='known difference at relocated vertices; full support remains valid'
    return z,g,meta
