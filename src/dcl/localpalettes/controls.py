"""Known-law statistical controls and isolated teacher-precision measurements."""
import numpy as np

from .adapter import converted_uniform
from .config import RESULTS, seed
from .report import write_csv
from .statistics import binary_product_tv, likelihood_curve, wilson


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


if __name__=='__main__':run_controls()
