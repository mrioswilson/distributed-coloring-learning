"""Execute the frozen acceptance rule; never weaken a failed target."""
import numpy as np
import argparse
import pandas as pd
from scipy.stats import binomtest
from .common import *
from .stats import holm

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,default=2);args=parser.parse_args();suffix='2' if args.round==2 else ''
    freeze=read(OUT/f'validation-freeze{suffix}.json')
    for name,digest in freeze['source_hashes'].items():
        if Path(name).name in ('stats.py','controls.py','controls_v2.py','classifier.py'):
            target=Path(name) if args.round==2 or Path(name).name!='stats.py' else Path(name).with_name('stats_v1.py')
            assert file_hash(target)==digest,str(target)
    v=pd.read_csv(OUT/f'benchmark-validation{suffix}.csv');nulls=v[v['case'].str.endswith('_null')]
    actual=pd.read_csv(OUT/f'actual-null-validation{suffix}.csv');cells=len(nulls)+len(actual);cut=.01/cells
    calibration=bool((nulls.null_exceedance_p>=cut).all() and (actual.null_exceedance_p>=cut).all())
    target_cells=[('localized',250),('localized_4',500),('correlation',1000),('higher_order',1000),('diffuse',1000)]
    if args.round==2:target_cells.append(('correlation_remote',1000))
    targets=[]
    for case,m in target_cells:
        a=v[(v['case']==case)&(v.m==m)&(v.method=='omnibus')].iloc[0]
        targets.append(dict(case=case,m=m,trials=int(a.trials),power=float(a.rate),ci_low=float(a.ci_low),passed=bool(a.ci_low>=.8)))
    d=v[(v['case']=='diffuse')&(v.m==1000)]
    global_loss=float(d[d.method=='old_hamming'].rate.iloc[0]-d[d.method=='omnibus'].rate.iloc[0])
    frame=pd.read_csv(OUT/f'benchmark-validation{suffix}-trials.csv');comparisons=[]
    for (case,m),a in frame[~frame['case'].str.endswith('_null')].groupby(['case','m']):
        w=a.pivot(index='trial',columns='method',values='reject').astype(bool);wins=int((w.omnibus&~w.old_hamming).sum());losses=int((~w.omnibus&w.old_hamming).sum())
        comparisons.append(dict(case=case,m=int(m),trials=len(w),wins=wins,losses=losses,paired_gain=(wins-losses)/len(w),mcnemar_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.))
    adj=holm([r['mcnemar_p'] for r in comparisons])
    for r,p in zip(comparisons,adj):r['mcnemar_holm']=float(p)
    pd.DataFrame(comparisons).to_csv(OUT/('paired-power-comparison.csv' if args.round==2 else 'paired-power-comparison-v1.csv'),index=False)
    passed=calibration and all(r['passed'] for r in targets) and global_loss<=.15
    tail=[]
    if args.round==2:
        tail=read(OUT/'tail-validation.json')
        passed=passed and all(r['ci_low']>=.8 for r in tail if r['m']==500)
    result=dict(passed=bool(passed),method='honest joint-permutation omnibus',calibration_passed=calibration,calibration_cells=cells,
      null_binomial_cut=cut,minimum_null_exceedance_p=float(min(nulls.null_exceedance_p.min(),actual.null_exceedance_p.min())),power_targets=targets,
      global_observed_power_loss=global_loss,tail_validation=tail,source_hashes=freeze['source_hashes'],note='Targets and method fixed before independent validation; nominal-alpha sensitivity on named alternatives and a separate strict-tail check. No universal power claim.')
    atomic_json(OUT/('instrument-acceptance.json' if args.round==2 else 'instrument-acceptance-v1.json'),result);print({k:v for k,v in result.items() if k!='source_hashes'},flush=True)
    if not passed:raise RuntimeError('Instrument acceptance failed; no confirmation allowed')
if __name__=='__main__':main()
