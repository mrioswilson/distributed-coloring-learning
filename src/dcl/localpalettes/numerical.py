"""Secondary execution-precision probe, with input rounding held fixed."""
import copy
import json
import numpy as np
import torch

from dcl.paperstudy.training import load_model
from .adapter import choices, converted_uniform
from .cohort import state_specs
from .config import RESULTS
from .evaluation import center_cache
from .report import write_csv


def run():
    torch.set_num_threads(1);rows=[]
    grid=(np.arange(257)+.5)/257
    states=[s for s in state_specs() if s['k'] in (4,51,201,501) and s['pattern'] in ('ALL_FREE','SINGLE-middle')]
    for spec in json.loads((RESULTS/'frozen-checkpoints.json').read_text()):
        if spec['seed']!=41000:continue
        model,_=load_model(spec['checkpoint'],'cpu');double=copy.deepcopy(model).double()
        for state in states:
            with torch.inference_mode():
                f32=center_cache(model,state);f64=center_cache(double,state)
                rc=converted_uniform(grid)
                a=choices(model,f32,rc);b=choices(double,f64,rc.astype(np.float64));c=choices(double,f64,grid)
            rows.append(dict(model=spec['label'],state=state['id'],k=state['k'],points=len(grid),
                             same_input_f32_f64_candidate_disagreement=float(np.mean(a!=b)),
                             f64_input_conversion_disagreement=float(np.mean(b!=c)),
                             f32_alpha_zero=int((f32[1]==0).sum()),f32_alpha_one=int((f32[1]==1).sum()),
                             f64_alpha_zero=int((f64[1]==0).sum()),f64_alpha_one=int((f64[1]==1).sum()),
                             f32_forbidden=float(np.mean(~np.isin(a,state['available']))),
                             f64_forbidden=float(np.mean(~np.isin(b,state['available']))),
                             note='Exploratory deterministic CPU grid; no probability confidence interval; weights unchanged'))
            print(spec['label'],state['id'],'precision disagreement',float(np.mean(a!=b)),flush=True)
    write_csv(RESULTS/'neural-precision-probes.csv',rows)


if __name__=='__main__':run()
