"""A graph-independent list-capacity bound for prefix local palettes."""
import numpy as np
import pandas as pd
from .common import *

def bound(degrees):
    d=np.asarray(degrees);K=int(d.max())+1
    # For any t, at most t colors below t can appear, and each vertex with
    # d(v)>=t supplies at most one used color above/equal t.
    values=[t+int((d>=t).sum()) for t in range(K+1)]
    t=int(np.argmin(values))
    return dict(used_color_upper_bound=values[t],minimizing_threshold=t,
                A_deterministically_true=bool(values[t]<=K-1))

def main():
    rows=[]
    for g in graphs():
        rows.append({k:g[k] for k in ('id','n','delta','stage','family')}|bound(g['degrees']))
    atomic_json(OUT/'palette-capacity-diagnostic.json',rows)
    pd.DataFrame(rows).to_csv(OUT/'palette-capacity-diagnostic.csv',index=False)
    print(pd.DataFrame(rows)[lambda a:a['stage']=='secondary'].to_string(index=False),flush=True)

if __name__=='__main__':main()
