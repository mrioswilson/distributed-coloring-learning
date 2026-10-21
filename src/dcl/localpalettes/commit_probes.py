"""Exploratory COMMIT diagnoses: actual legal local-list stars, no training."""
import json
import numpy as np
import torch

from dcl.paperstudy.training import load_model
from .adapter import LocalPaletteAdapter
from .cohort import star_edges
from .config import RESULTS
from .report import write_csv


def run():
    torch.set_num_threads(1);rows=[]
    for spec in json.loads((RESULTS/'frozen-checkpoints.json').read_text()):
        model,_=load_model(spec['checkpoint'],'cpu');adapter=LocalPaletteAdapter(model)
        for d in (0,1,2,3,4,8,16,50,200,500):
            for mode in ('FIXED_OTHER','ONE_FIXED_SAME','PENDING_OTHER','ONE_PENDING_SAME'):
                if d==0 and mode.startswith('ONE_'):continue
                colors=np.full(d+1,-1,dtype=np.int64);part=np.ones(d+1,dtype=bool);candidate=np.ones(d+1,dtype=np.int64);candidate[0]=0
                if 'FIXED' in mode:
                    colors[1:]=1;part[1:]=False;candidate[1:]=-1
                if mode=='ONE_FIXED_SAME':colors[1]=0
                if mode=='ONE_PENDING_SAME':candidate[1]=0
                palette=np.r_[d+1,np.full(d,2)]
                assert np.all(colors<palette) and np.all(candidate<palette)
                with torch.inference_mode():
                    logit=float(adapter.commit_logits(torch.tensor(colors),torch.tensor(star_edges(d+1)),torch.tensor(palette),
                                                      torch.tensor(part),torch.tensor(candidate))[0])
                exact=not mode.startswith('ONE_')
                rows.append(dict(model=spec['label'],treatment=spec['treatment'],seed=spec['seed'],degree=d,mode=mode,
                                 logit=logit,exact_accept=exact,learned_accept=logit>0,correct=(logit>0)==exact,
                                 kind='exploratory-conditional-COMMIT-intervention',checkpoint_sha256=spec['sha256']))
    write_csv(RESULTS/'commit-degree-probes.csv',rows)
    print(f'Exploratory local-list COMMIT probes: {len(rows)}.')


if __name__=='__main__':run()
