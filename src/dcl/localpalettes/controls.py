"""Known-law statistical controls and isolated teacher-precision measurements."""
import numpy as np
from fractions import Fraction
import json
import torch

from .adapter import converted_uniform
from .config import RESULTS, seed
from .report import write_csv
from .statistics import binary_product_tv, likelihood_curve, wilson


def participation_measure(head):
    """Exact rational integration of this two-layer ReLU head at uncolored=1.

    Float32 coefficients are interpreted as exact real rationals. This is not
    a statement about rounded intermediate float32 arithmetic.
    """
    f=lambda value:Fraction(float(value))
    a=[f(v) for v in head[0].weight[:,0].detach().cpu()]
    b=[f(w)+f(c) for w,c in zip(head[0].weight[:,1].detach().cpu(),head[0].bias.detach().cpu())]
    w=[f(v) for v in head[2].weight[0].detach().cpu()]
    bias=f(head[2].bias[0].detach().cpu())
    knots=sorted({Fraction(0),Fraction(1)}|{-bj/aj for aj,bj in zip(a,b) if aj and 0 < -bj/aj < 1})
    mass=Fraction(0);positive=[]
    for left,right in zip(knots[:-1],knots[1:]):
        middle=(left+right)/2
        active=[j for j in range(len(a)) if a[j]*middle+b[j]>0]
        slope=sum((w[j]*a[j] for j in active),Fraction(0))
        intercept=bias+sum((w[j]*b[j] for j in active),Fraction(0))
        pieces=[left,right]
        if slope:
            root=-intercept/slope
            if left < root < right:pieces.insert(1,root)
        for lo,hi in zip(pieces[:-1],pieces[1:]):
            if slope*(lo+hi)/2+intercept>0:
                mass+=hi-lo;positive.append([float(lo),float(hi)])
    return dict(probability_real=float(mass),exact_rational=str(mass),positive_intervals=positive,
                hidden_breakpoints=len(knots)-2,interpretation='Exact real evaluation of frozen coefficients; not a float32 arithmetic certificate')


def participation_controls():
    from dcl.paperstudy.training import load_model
    from dcl.universal.train import load_model as old_load
    from dcl.study.storage import atomic_json
    torch.set_num_threads(1)
    specs=json.loads((RESULTS/'frozen-checkpoints.json').read_text())+json.loads((RESULTS/'historical-e1-checkpoints.json').read_text())
    rows=[]
    for spec in specs:
        model,_=(old_load if spec.get('kind')=='historical-e1' else load_model)(spec['checkpoint'],'cpu')
        rows.append(dict(model=spec['label'],checkpoint_sha256=spec['sha256'],**participation_measure(model.participation)))
    atomic_json(RESULTS/'participation-exact-real.json',rows)
    print(f'Exact rational participation integration: {len(rows)} frozen heads.')


def run_controls():
    rows=[]
    for name,p in (('identical-uniform-shift',.5),('biased-binary',.8),('point-mass',1.)):
        est=np.random.default_rng(seed('known-law',name,'estimate')).choice(2,32768,p=[1-p,p])
        alt=np.random.default_rng(seed('known-law',name,'test-alt')).choice(2,(256,2000),p=[1-p,p])
        null=np.random.default_rng(seed('known-law',name,'test-null')).integers(0,2,(256,2000))
        curve=likelihood_curve(np.bincount(est,minlength=2),np.arange(2),alt,null,
                              np.random.default_rng(seed('known-law',name,'calibration')),
                              tie_rng=np.random.default_rng(seed('known-law',name,'ties')))
        for r in curve:
            tv=binary_product_tv(.5,p,r['m'])
            rows.append(dict(control=name,true_product_tv=tv,true_bayes_accuracy=.5*(1+tv),**r))
    write_csv(RESULTS/'known-law-controls.csv',rows)
    precision=[]
    for k in (1,2,3,4,7,11,51,101,201,501,997):
        r=np.random.default_rng(seed('teacher-precision',k)).random(1_000_000)
        before=np.floor(r*k).astype(int);after=np.floor(converted_uniform(r).astype(float)*k).astype(int)
        difference=int(np.sum(before!=after))
        boundary=(np.arange(1,k,dtype=float)/k).astype(np.float32)
        grid=np.unique(np.r_[boundary,np.nextafter(boundary,np.float32(0)),np.nextafter(boundary,np.float32(1)),
                              np.float32(0),np.nextafter(np.float32(1),np.float32(0))])
        grid=grid[(grid>=0)&(grid<1)]
        numeric64=np.floor(grid.astype(float)*k).astype(int)
        numeric32=np.floor(grid*np.float32(k)).astype(int)
        precision.append(dict(available_colors=k,iid_samples=len(r),teacher_conversion_disagreements=difference,
                              disagreement_rate=difference/len(r),wilson_interval=wilson(difference,len(r)),
                              continuous_coupling_upper=min(1.,2*(k-1)*2**-24),
                              deterministic_boundary_points=len(grid),
                              float32_vs_float64_multiplication_disagreements=int(np.sum(numeric32!=numeric64)),
                              note='Boundary grid is deterministic; its discrepancy fraction is not a probability estimate'))
    write_csv(RESULTS/'teacher-precision.csv',precision)
    print(f'Known-law classifier controls: {len(rows)} rows. Teacher-precision contexts: {len(precision)}.')


if __name__=='__main__':
    run_controls()
    participation_controls()
