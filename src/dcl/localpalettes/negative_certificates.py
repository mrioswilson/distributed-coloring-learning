"""Interval certificates of concrete COMMIT errors, not universal impossibility."""
import json
import copy
import numpy as np
import torch

from dcl.paperstudy.certification import TYPES, down, up, module_interval
from dcl.paperstudy.training import load_model
from dcl.study.storage import atomic_json
from .config import RESULTS
from .cohort import star_edges
from .adapter import LocalPaletteAdapter


def run():
    rows=[]
    for spec in json.loads((RESULTS/'frozen-checkpoints.json').read_text()):
        treatment=spec['treatment']
        if treatment not in ('sum','mean'):continue
        model,_=load_model(spec['checkpoint'],'cpu');double=copy.deepcopy(model).double()
        cases=[(8,{4:8},True,'all-pending-other')] if treatment=='sum' else [
              (4,{0:3,1:1},False,'one-fixed-same'),(8,{4:7,5:1},False,'one-pending-same'),
              (8,{0:7,1:1},False,'one-fixed-same'),(16,{4:15,5:1},False,'one-pending-same')]
        for degree,counts,should_accept,name in cases:
            for float32 in (False,True):
                lo,hi=module_interval(model.message2,TYPES,TYPES,float32)
                rlo,rhi=np.zeros(32),np.zeros(32)
                for typ,count in counts.items():
                    rlo=down(rlo+down(count*lo[typ]));rhi=up(rhi+up(count*hi[typ]))
                if treatment=='mean':rlo,rhi=down(rlo/degree),up(rhi/degree)
                if float32:
                    unit=2**-24;terms=degree+2
                    gamma=up(terms*unit/(1-terms*unit))
                    error=up(gamma*np.maximum(np.abs(rlo),np.abs(rhi))+terms*2**-149)
                    rlo,rhi=np.maximum(0,down(rlo-error)),up(rhi+error)
                center=np.array([1.,0.,1.,1.])
                zlo,zhi=module_interval(model.state2,np.r_[center,rlo],np.r_[center,rhi],float32)
                clo,chi=module_interval(model.commit_head,np.r_[zlo,1,1,1],np.r_[zhi,1,1,1],float32)
                lower,upper=float(clo[0]),float(chi[0])
                error_certified=upper<0 if should_accept else lower>0
                q=[-1];p=[True];candidate=[0]
                for typ,count in counts.items():
                    for _ in range(count):
                        q.append(1 if typ==0 else 0 if typ==1 else -1)
                        p.append(typ in (4,5));candidate.append(0 if typ==5 else 1 if typ==4 else -1)
                selected_model=model if float32 else double
                with torch.inference_mode():
                    pathway=selected_model.commit_logits if float32 else LocalPaletteAdapter(selected_model).commit_logits
                    actual=float(pathway(torch.tensor(q),torch.tensor(star_edges(degree+1)),
                                 torch.tensor([degree+1]+[2]*degree),torch.tensor(p),torch.tensor(candidate))[0])
                assert lower<=actual<=upper,(spec['label'],name,degree,lower,actual,upper)
                rows.append(dict(model=spec['label'],checkpoint_sha256=spec['sha256'],degree=degree,
                                 center_local_palette=degree+1,leaf_local_palette=2,case=name,expected_accept=should_accept,
                                 logit_lower=lower,logit_upper=upper,certified_error=error_certified,
                                 certified_correct=lower>0 if should_accept else upper<0,
                                 actual_validation_logit=actual,inside_interval=True,
                                 arithmetic='conditional-float32-error-model' if float32 else 'exact-real-coefficients',
                                 assumptions=['legal local-list star and supplied proposals','selected candidate 0',
                                              'exact real float32 coefficients','outward float64 interval enclosures']+
                                 (['IEEE nonnegative aggregation gamma_(d+2)','affine gamma_(n+2)','no TF32/overflow'] if float32 else [])))
    atomic_json(RESULTS/'negative-commit-certificates.json',rows)
    print('Certified concrete errors:',sum(r['certified_error'] for r in rows),'/',len(rows))


if __name__=='__main__':run()
