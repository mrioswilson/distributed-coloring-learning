"""Separate simultaneous effect bounds using the original honest witnesses.

No original-policy test, witness, intervention action or sample is changed.
This supplementary family was specified after original witness examples were
seen and before any intervention observation. All sixty cells are retained.
"""
import argparse
import subprocess
import numpy as np
import pandas as pd
from .common import *
from .intervention import IROOT,IOUT,target_graphs
from dcl.robustdist.classifier import interval

DOC=Path('docs/FACTORIAL_INTERVENTION_WITNESS_ADDENDUM.md')

def effect_bounds(kx,nx,ky,ny,kdiff,m,success_x,success_y,alpha):
    cx=interval(kx,nx,alpha) if nx else [0.,1.]
    cy=interval(ky,ny,alpha) if ny else [0.,1.]
    diff=[cy[0]-cx[1],cy[1]-cx[0]]
    original=max(0.,diff[0],-diff[1])
    coupling=interval(kdiff,m,alpha)
    sx=interval(success_x,m,alpha);sy=interval(success_y,m,alpha)
    upper=min(1.,2*coupling[1]/max(min(sx[0],sy[0]),1e-300))
    return dict(original_witness_difference_ci=diff,original_witness_tv_lower=original,
                conditional_tv_upper=upper,intervention_tv_effect_lower=max(0.,original-upper),
                mismatch_probability_ci=coupling,success_ci_Luby=sx,success_ci_intervened=sy,
                original_probability_ci_Luby=cx,original_probability_ci_model=cy)

def plan():
    verify_frozen()
    existing=list((IROOT/'banks').glob('*.npz'))
    if existing:raise RuntimeError('Supplement must be registered before intervention observations')
    record=dict(created=now(),preexisting_intervention_banks=0,
                original_protocol_commit=read(OUT/'protocol-commit.json')['commit'],
                intervention_protocol_commit=read(IOUT/'protocol-commit.json')['commit'],
                cells=60,probabilities_per_cell=5,confidence=.95,
                alpha_per_probability=.05/(60*5),
                graphs=[g['id'] for g in target_graphs()],checkpoints=[s['label'] for s in specs()],
                source_sha256=file_hash(__file__),document_sha256=file_hash(DOC),
                scope='Exploratory separate family; specified after original holdout examples, before any intervention observation')
    atomic_json(IOUT/'witness-addendum.json',record)

def analyze():
    verify_frozen();p=read(IOUT/'witness-addendum.json')
    assert p['preexisting_intervention_banks']==0
    assert file_hash(__file__)==p['source_sha256'] and file_hash(DOC)==p['document_sha256']
    commit=read(IOUT/'witness-addendum-commit.json')['commit']
    assert subprocess.check_output(['git','show',commit+':src/dcl/sizedegree/intervention_witnesses.py'])==Path(__file__).read_bytes()
    completion=read(IOUT/'completion.json');queue=IROOT/'queue.json';rows=[]
    if queue.exists():
        original={}
        for stratum in ('main','historical-replication'):
            original.update({(r['graph'],r['model']):r for r in read(OUT/f'{stratum}.json')})
        for t in read(queue)['tasks']:
            if t['status']!='complete':continue
            assert t['graph'] in p['graphs'] and t['model']['label'] in p['checkpoints']
            assert file_hash(t['file'])==t['sha256']
            r=original[(t['graph'],t['model']['label'])];w=r.get('witness')
            if w is None:continue
            with np.load(t['file']) as a:
                b=effect_bounds(w['count_Luby'],w['n_Luby'],w['count_model'],w['n_model'],
                                int(a['coarsened_mismatch'].sum()),len(a['seeds']),
                                int((a['status_Luby']==0).sum()),int((a['status_intervened']==0).sum()),
                                p['alpha_per_probability'])
            rows.append(dict(graph=t['graph'],model=t['model']['label'],architecture=t['model']['treatment'],
                             pairs=t['m'],event=w['event'],probability_Luby=w['probability_Luby'],
                             probability_model=w['probability_model'],
                             count_Luby=w['count_Luby'],count_model=w['count_model'],
                             holdout_Luby=w['n_Luby'],holdout_model=w['n_model'],**b))
    result=dict(status=completion['status'],cells_planned=60,cells_completed=len(rows),
                simultaneous_confidence=.95,elementary_probabilities=300,rows=rows,
                effect_positive=sum(r['intervention_tv_effect_lower']>0 for r in rows),
                scope=p['scope'],addendum_commit=commit,
                warning='Separate 95% family: no joint 95% guarantee over its union with the original event-effect family')
    atomic_json(IOUT/'witness-effects.json',result);pd.DataFrame(rows).to_csv(IOUT/'witness-effects.csv',index=False)
    print(json.dumps(dict(cells=len(rows),positive_effect_bounds=result['effect_positive'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','analyze']);a=p.parse_args()
    {'plan':plan,'analyze':analyze}[a.phase]()

if __name__=='__main__':main()
