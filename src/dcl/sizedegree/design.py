"""Freeze by costs and independent instrument checks, never neural p-values."""
import argparse
import subprocess
import numpy as np
from .common import *
from .graphs import validate
from .inference import contrast_specs,EVENTS

def history():
    gid='regular-n512-d100';label='normalized-s41000';items=[]
    for who in ('Luby',label):
        path=BASE/'data/ucgnn-robust-distribution/banks'/f'confirmatory__{who}__{gid}.npz'
        with np.load(path) as a:
            X=a['colors'][-250:][a['status'][-250:]==0]
            k=sum(len(set(x))<=100 for x in X)
        items.append(dict(generator=who,count=int(k),total=len(X),probability=k/len(X),source=str(path),sha256=file_hash(path)))
    old=read(BASE/'results/ucgnn-robust-distribution/explanatory-witnesses.json')
    r=next(r for r in old if r['graph']==gid and r['model']==label)
    assert [r['probability_Luby'],r['probability_model']]==[v['probability'] for v in items]
    result=dict(reproduced_exactly=True,historical_data_only=True,event='used_colors <= 100',banks=items,
                original_difference_ci=r['difference_ci'],original_event=r['event'])
    atomic_json(OUT/'historical-reproduction.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    if (OUT/'freeze.json').exists():raise RuntimeError('Already frozen')
    q=read(ROOT/'queue-cost.json')
    if not all(t['status']=='complete' for t in q['tasks']):raise RuntimeError('Cost pilot incomplete')
    gs=read(OUT/'candidate-graphs.json');cg={g['id']:g for g in read(OUT/'cost-graphs.json')}
    costs={}
    for t in q['tasks']:
        label='Luby' if t['model'] is None else t['model']['treatment'];g=cg[t['graph']]
        costs[(label,g['family'],g['n'],g['delta'])]=t['seconds']/t['m']
    budget=read(ROOT/'budget.json');reserve=900.;forecasts=[]
    # All candidate reductions are balanced Cartesian products and use costs only.
    options=[('full16',NS,DS),('eight', (128,1024),DS),('four',(128,1024),(10,100))]
    chosen=None
    for name,ns,ds in options:
        keep=[g for g in gs if g['stage']!='secondary' or (g['n'] in ns and g['delta'] in ds)]
        total=0.;breakdown={}
        for g in keep:
            for label in ('Luby','normalized','deepsets','transformer'):
                amount=costs[(label,g['family'],g['n'],g['delta'])]*625*(1 if label=='Luby' else 5)
                total+=amount;breakdown[label]=breakdown.get(label,0.)+amount
        protected=budget['seconds']+1.25*total+reserve
        forecasts.append(dict(option=name,graphs=len(keep),predicted_confirmation_seconds=total,
                              protected_total_seconds=protected,by_generator_seconds=breakdown))
        if chosen is None and protected<=43200:
            chosen=(name,keep,forecasts[-1])
    atomic_json(OUT/'resource-design-candidates.json',forecasts)
    if chosen is None:raise RuntimeError('Even balanced four-cell secondary design does not fit: revise prospectively before freezing')
    name,keep,cost=chosen
    atomic_json(OUT/'graphs.json',keep);atomic_json(OUT/'graph-audit.json',validate(keep+read(OUT/'pilot-graphs.json')+[g for g in cg.values() if g['family']=='irregular']))
    history()
    families={s:sum(g['stage']==s for g in keep)*15 for s in ('main','secondary','historical-replication')}
    f=dict(created=now(),base_commit='74134d865db4f142bfeea49b091200588d1f2803',method='unchanged frozen omnibus V2',
           graphs_per_cell=3,test_runs_per_generator=500,explanation_runs_per_generator=125,total_runs_per_bank=625,
           permutations=19999,alpha=.05,hypothesis_families=families,total_comparisons=sum(families.values()),
           elementary_event_probabilities=len(keep)*16*len(EVENTS),color_probabilities=sum((g['delta']+1)*16 for g in keep),
           factorial_contrasts=len(contrast_specs(NS,DS))*len(EVENTS)*18,
           selected_secondary_option=name,resource_forecast=cost,mechanism_reserve_seconds=reserve,
           graphs_sha256=file_hash(OUT/'graphs.json'),checkpoints_sha256=file_hash(OUT/'checkpoints.json'))
    assert 1/(f['permutations']+1)<=.05/max(families.values())
    for s in specs():assert file_hash(s['checkpoint'])==s['sha256']
    if not args.freeze:
        atomic_json(OUT/'prospective-design.json',f);print(json.dumps(f,indent=2));return
    if not read(OUT/'validation.json')['passed']:raise RuntimeError('Validation failed')
    import xml.etree.ElementTree as ET
    suites=ET.parse(OUT/'preflight-tests.xml').getroot()
    assert sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites.iter('testsuite'))==0
    paths=[Path('src/dcl/sizedegree')/f'{name}.py' for name in ('common','graphs','inference','run','analysis')]
    paths += [Path(p) for p in read(BASE/'results/ucgnn-robust-distribution/freeze.json')['source_hashes']]
    # The unchanged witness inference helpers also enter the frozen dependency closure.
    paths += [Path('src/dcl/robustdist/numerics.py')]
    f['source_hashes']={str(p):file_hash(p) for p in paths}
    f['protocol_sha256']=file_hash('docs/FACTORIAL_SIZE_DEGREE_PROTOCOL.md')
    f['validation_sha256']=file_hash(OUT/'validation.json')
    atomic_json(OUT/'freeze.json',f)
    print(json.dumps(dict(frozen=True,design=name,graphs=len(keep),families=families,forecast_hours=cost['protected_total_seconds']/3600)))

if __name__=='__main__':main()
