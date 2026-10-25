#!/usr/bin/env python3
"""Per-checkpoint diagnostic denominators and paired update-matched contrasts."""
from collections import defaultdict
import csv
import json
import numpy as np
from dcl.degree.inventory import ROOT
from dcl.degree.report import write_csv, table
from dcl.study.storage import atomic_json


if __name__ == "__main__":
    results=json.loads((ROOT/'counterfactual-probes/index.json').read_text())
    rows=[]
    for r in results:
        for c,q in r['commit_queries'].items():
            rows.append(dict(checkpoint=r['checkpoint'],teacher=r['teacher'],architecture=r['architecture'],
                cohort=r['cohort'],method=r['method'],update=r['update'],seed=r['seed'],condition=c,
                **q,accuracy=q['correct']/q['total']))
    write_csv(ROOT/'counterfactual-probes/checkpoint-metrics.csv',rows)
    pairs=[]
    for cf in results:
        if cf['method']!='counterfactual':continue
        for method in ('matched','passive'):
            controls=[r for r in results if r['teacher']==cf['teacher'] and r['architecture']==cf['architecture']
                and r['seed']==cf['seed'] and r['method']==method and r['update']==cf['update']]
            assert len(controls)==1
            control=controls[0]
            for condition in cf['commit_queries']:
                a,b=cf['commit_queries'][condition],control['commit_queries'][condition]
                assert a['total']==b['total'] and a['positive']==b['positive'] and a['negative']==b['negative']
                pairs.append(dict(teacher=cf['teacher'],architecture=cf['architecture'],seed=cf['seed'],
                    updates=cf['update'],control=method,condition=condition,total=a['total'],
                    cf_correct=a['correct'],control_correct=b['correct'],
                    accuracy_difference=(a['correct']-b['correct'])/a['total']))
    write_csv(ROOT/'counterfactual-probes/paired-equal-update.csv',pairs)
    gap_rows=[]
    for r in results:
        for x in r['preference_gap_probes']:
            gap_rows.append(dict(checkpoint=r['checkpoint'],teacher=r['teacher'],architecture=r['architecture'],
                method=r['method'],cohort=r['cohort'],seed=r['seed'],**x))
    write_csv(ROOT/'counterfactual-probes/preference-gap.csv',gap_rows)
    primary=[r for r in gap_rows if r['cohort']=='counterfactual-primary' and r['palette']=='local' and r['gap']==1]
    stats=[]
    for d in (3,4,5,10,20,50,100,200):
        xs=[r for r in primary if r['delta']==d]
        stats.append(dict(delta=d,endpoint_degree=1,endpoint_palette=2,checkpoints=len(xs),
            exact_winner_rejected=sum(not r['forced_zero_commit'] for r in xs),
            min_forced_logit=min(r['forced_zero_logit'] for r in xs),max_forced_logit=max(r['forced_zero_logit'] for r in xs)))
    atomic_json(ROOT/'counterfactual-probes/mechanism-summary.json',stats)
    print(json.dumps(stats))
