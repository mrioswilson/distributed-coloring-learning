"""Instrument results written before any fresh teacher/network comparison."""
import numpy as np
import pandas as pd
from .common import *

def generate():
    v=pd.read_csv(OUT/'benchmark-validation2.csv');a=read(OUT/'instrument-acceptance.json')
    assert a['passed'] and not (DATA/'queue-confirmatory.json').exists()
    actual=pd.read_csv(OUT/'actual-null-validation2.csv');null=v[v['case'].str.endswith('_null')].copy();null['law']=null['case'];actual['law']='actual '+actual.label
    n=pd.concat([null,actual],ignore_index=True)
    n.to_csv(OUT/'instrument-null-calibration.csv',index=False)
    methods=['old_hamming','multi_max','multi_softmax','omnibus_v1','omnibus','local','interaction','structural','split_multiscale','split_global','classifier','multinomial']
    text=['# Independent instrument validation before neural confirmation','',
      'This artifact is generated before the first new teacher-versus-network bank. It documents method choice, acceptance and limitations separately from later scientific results. V1, V2, development and validation seed namespaces remain distinct.','',
      'The final method is the honestly split joint-permutation omnibus V2. Its training-only all-pair search was motivated by separate relocated-pair development failures. All V2 parameters were fixed before second-round validation. No new neural comparison was used to select a kernel, region, checkpoint, sample size or test.','',
      '## Calibration','',f'There are {a["calibration_cells"]} method/null cells. Each null law has 400 independent trials at total m=250 per generator. The prespecified family audit uses one-sided binomial exceedance p >= {a["null_binomial_cut"]:.8g}. Minimum observed audit p is {a["minimum_null_exceedance_p"]:.8g}. Passing finite controls is not a universal guarantee; conditional exchangeability and joint permutation ranks provide the level argument. Pointwise intervals may exclude .05 by chance among many cells; the multiplicity-aware audit is reported without hiding them.','',
      '| Null law | Omnibus rejections/400 | Wilson 95% interval |','|---|---:|---|']
    for r in n[n.method=='omnibus'].itertuples():text.append(f'| {r.law} | {r.rejections}/400 | [{r.ci_low:.3f}, {r.ci_high:.3f}] |')
    text+=['','instrument-null-calibration.csv and both raw validation tables retain every method; point-mass, synthetic, actual Luby and actual frozen-checkpoint controls are distinguished. All neural self controls use fresh executions, not resampling fitted empirical probabilities.','',
      '## Fixed power targets','', '| Class | Total m per generator | Detections/30 | Lower pointwise Wilson 95% bound |','|---|---:|---:|---:|']
    for t in a['power_targets']:
        row=v[(v['case']==t['case'])&(v.m==t['m'])&(v.method=='omnibus')].iloc[0]
        text.append(f'| {t["case"]} | {t["m"]} | {int(row.rejections)}/30 | {t["ci_low"]:.3f} |')
    text+=['','Selection observations are counted in total m. Baselines use all m; split methods reserve half for independent evaluation. These targets concern specified alternatives, not every law with the same TV. Thirty trials have substantial uncertainty.','',
      '## All candidate methods on the mandatory star control','', '| Method | m=100 | 250 | 500 | 1000 | 2000 |','|---|---:|---:|---:|---:|---:|']
    for method in methods:
        cells=[]
        for m in (100,250,500,1000,2000):
            row=v[(v['case']=='localized')&(v.method==method)&(v.m==m)]
            cells.append(f'{int(row.rejections.iloc[0])}/30' if len(row) else 'not run')
        text.append('| '+method+' | '+' | '.join(cells)+' |')
    text+=['','The star laws are artificial proper-coloring generators, not the actual Luby and neural output laws. Their known TV is 99/199. The old Hamming MMD has population squared effect 1.50213168648e-5 and null SD .000241288 at m=250. The simple center event <=101 has probabilities 100/199 and 1. This exhibits kernel attenuation rather than an information-theoretic inability to distinguish those two known laws.','',
      '## Global and higher-order sensitivity, with failures retained','', '| Class | Old MMD at m=1000 | V1 at m=1000 | V2 at m=1000 |','|---|---:|---:|---:|']
    for case in ('marginal','localized_4','correlation','correlation_remote','higher_order','higher_order_remote','parity_global','structural','diffuse','large_support','rare'):
        cells=[]
        for method in ('old_hamming','omnibus_v1','omnibus'):
            row=v[(v['case']==case)&(v.method==method)&(v.m==1000)].iloc[0];cells.append(f'{int(row.rejections)}/30 [{row.ci_low:.2f},{row.ci_high:.2f}]')
        text.append('| '+case+' | '+' | '.join(cells)+' |')
    text+=['','Unknown relocated triples are not covered by an all-pair search. The retained characteristic component ensures fixed-space consistency in ideal arithmetic, not adequate finite power for these high-order alternatives. Rare-event sampling and huge unknown supports give explicit lower bounds; no 80% target is imposed on statistically infeasible cases. Structural summaries are not injective and do not replace the complete-vector component.','',
      '## Primary-tail check','', '| Class | m | B | Threshold | Detections/30 | Wilson 95% interval |','|---|---:|---:|---:|---:|---|']
    for r in read(OUT/'tail-validation.json'):text.append(f'| {r["case"]} | {r["m"]} | 19999 | .05/795 | {r["detections"]}/30 | [{r["ci_low"]:.3f},{r["ci_high"]:.3f}] |')
    text+=['','The two total-m=500 cells passed the separately fixed >=.80 lower-bound gate. These are conservative first-Holm-threshold checks for named laws, not estimates of power under the entire unknown neural family.','',
      '## Compute cost and reproducibility','', '| Total m | Mean omnibus seconds | Mean combined four-global-kernel seconds |','|---:|---:|---:|']
    trials=pd.read_csv(OUT/'benchmark-validation2-trials.csv')
    for m in (100,250,500,1000,2000):
        d=trials[trials.m==m];text.append(f'| {m} | {d.omnibus_seconds.mean():.4f} | {d.quadratic_seconds.mean():.4f} |')
    text+=['','Timing columns measure complete omnibus and combined global-scale calculations; they do not estimate an isolated single-kernel runtime. CPU worker contention is part of measured wall time. Algorithm execution costs have a separate GPU-process ledger and prospective per-cell budget matrix. No neural network was trained. The simple source classifiers are statistical tools trained only within their designated blocks.','',
      'The complete validation arrays and hashes, trial-level p-values, seeds and both method versions are archived. All sample sizes, bandwidths, feature families and external multiplicity rules for new model comparisons will be frozen in the separate confirmation protocol. MMDAgg and MMD-FUSE motivated comparisons but are not claimed as exact reproduced algorithms or inherited minimax guarantees.']
    (OUT/'INSTRUMENT_VALIDATION_REPORT.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':generate()
