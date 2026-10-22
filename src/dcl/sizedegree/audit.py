"""Independent end-to-end audit of the archived factorial experiment."""
import argparse
from datetime import datetime
import json
import subprocess
import numpy as np
import pandas as pd
from .common import *
from .graphs import validate
from .inference import measurements,EVENTS
from dcl.globaldist.core import seed as run_seed,holm
from dcl.globaldist.sampling import sample
from dcl.robustdist.stats import joint_test
from dcl.robustdist.confirmation import event_values

def main():
    f=verify_frozen();record=read(OUT/'protocol-commit.json');commit=record['commit']
    frozen_doc=subprocess.check_output(['git','show',commit+':docs/FACTORIAL_SIZE_DEGREE_PROTOCOL.md'])
    import hashlib
    assert hashlib.sha256(frozen_doc).hexdigest()==f['protocol_sha256']
    prereg=datetime.fromisoformat(subprocess.check_output(['git','show','-s','--format=%cI',commit],text=True).strip())
    assert record['confirmation_banks_before_freeze']==0
    assert file_hash(OUT/'validation.json')==f['validation_sha256']
    assert read(OUT/'validation.json')['passed']
    for s in specs():assert file_hash(s['checkpoint'])==s['sha256']
    allgs=graphs()+read(OUT/'pilot-graphs.json')+[g for g in read(OUT/'cost-graphs.json') if g['family']=='irregular']
    graph_audit=validate(allgs);gs={g['id']:g for g in allgs}
    manifests=[];newseeds=[];allstatus=np.zeros(4,np.int64)
    for phase in ('pilot','cost','confirmation'):
        q=read(ROOT/f'queue-{phase}.json');assert all(t['status']=='complete' for t in q['tasks'])
        if phase=='confirmation':assert datetime.fromisoformat(q['created'])>prereg
        for t in q['tasks']:
            label='Luby' if t['model'] is None else t['model']['label'];g=gs[t['graph']]
            path=ROOT/'banks'/f'{t["id"]}.npz';assert file_hash(path)==t['sha256']
            a=bank(label,t['graph'],phase);x=a['colors'];status=a['status'];seeds=a['seeds'];newseeds.append(seeds)
            expected=np.array([run_seed(t['stage'],t['graph'],label,'run',i) for i in range(t['m'])],np.uint64)
            assert np.array_equal(seeds,expected)
            assert x.shape==(t['m'],g['n']) and np.isin(status,[0,1,2,3]).all()
            edges=np.array(g['edges']);valid=(x>=0).all(1)&(x<=np.array(g['degrees'])).all(1)&(x[:,edges[0]]!=x[:,edges[1]]).all(1)
            assert valid[status==0].all()
            assert not valid[status==1].any()
            assert (x[status==2]<0).any(1).all()
            assert (a['rounds'][status==2]==128).all()
            assert json.loads(str(a['first_rng_initial_state']))==np.random.default_rng(int(seeds[0])).bit_generator.state
            counts=np.bincount(status,minlength=4);assert counts.tolist()==t['statuses'];allstatus+=counts
            manifests.append(dict(phase=phase,id=t['id'],graph=t['graph'],model=label,file=str(path),sha256=t['sha256'],
                                  attempts=len(x),valid=int(counts[0]),invalid=int(counts[1]),cutoff=int(counts[2]),error=int(counts[3]),
                                  seconds=t['seconds'],max_rounds=int(a['rounds'].max())))
    mq=read(ROOT/'mechanism-queue.json')
    assert len(mq['complete'])==len(mq['tasks'])
    mechanism_errors=np.zeros(2,np.int64)
    for t in mq['tasks']:
        path=ROOT/'mechanism'/f'{t["id"]}.npz';assert file_hash(path)==t['sha256']
        with np.load(path) as a:
            newseeds.append(a['seeds']);mechanism_errors+=a['observer_errors'].sum((0,1))
            label='Luby' if t['model'] is None else t['model']['label'];g=gs[t['graph']]
            expected=[run_seed('factorial-v1/mechanism',g['id'],label,'run',i) for i in range(8)]
            assert np.array_equal(a['seeds'],np.array(expected,np.uint64))
            assert (a['commit_counts'].sum((0,2))==(a['colors']>=0).sum(1)).all()
        manifests.append(dict(phase='mechanism',id=t['id'],graph=t['graph'],model=label,file=str(path),sha256=t['sha256'],attempts=8,seconds=t['seconds']))
    assert mechanism_errors[1]==0
    new=np.concatenate(newseeds);assert len(np.unique(new))==len(new)
    oldseeds=[];oldbank_count=0
    for root,sub in [(GLOBAL,'ucgnn-global-distribution'),(BASE,'ucgnn-robust-distribution')]:
        for path in (root/'data'/sub/'banks').glob('*.npz'):
            with np.load(path) as a:
                if 'seeds' in a.files:oldseeds.append(a['seeds']);oldbank_count+=1
    old=np.concatenate(oldseeds);assert not len(np.intersect1d(new,old))
    auxiliary=np.array([seed('transfer',case,t,side) for case in ('teacher_empirical_null','neural_empirical_null','large_empirical_null','point_null','star_positive','diffuse_positive')
                        for t in range(200 if 'empirical_null' in case else (20 if case=='point_null' else 30)) for side in ('A','B','permutation','old')],np.uint64)
    assert len(np.unique(auxiliary))==len(auxiliary)
    assert not len(np.intersect1d(auxiliary,new)) and not len(np.intersect1d(auxiliary,old))
    # Fresh teacher-bank replay uses the unchanged sampler on CPU.
    g=next(g for g in graphs() if g['stage']=='main' and g['n']==128 and g['delta']==3 and g['rep']==0)
    a=bank('Luby',g['id']);replay=sample(g,None,'Luby','factorial-v1/confirmation',f['total_runs_per_bank'])
    for k in ('colors','status','rounds','seeds','first_rng_initial_state','final_rng_states_sha256'):
        assert np.array_equal(a[k],replay[k]),k
    tests=[];allrows=[]
    for stratum,N in f['hypothesis_families'].items():
        rows=read(OUT/f'{stratum}.json');assert len(rows)==N
        for key,adjusted in [('p','p_holm'),('old_p','old_p_holm'),('completion_p','completion_p_holm')]:
            assert np.array_equal(holm([r[key] for r in rows]),[r[adjusted] for r in rows])
        allrows.extend(rows)
    # Replay a rejection and a nonrejection; use its recorded full/futility route.
    for wanted in (True,False):
        r=next(r for r in allrows if r['stratum']=='main' and r['reject']==wanted)
        g=gs[r['graph']];a,b=bank('Luby',g['id']),bank(r['model'],g['id']);m=f['test_runs_per_generator']
        X=a['colors'][:m][a['status'][:m]==0];Y=b['colors'][:m][b['status'][:m]==0]
        q=joint_test(X,Y,g,rng('confirmation',g['id'],r['model'],'omnibus'),B=f['permutations'],
                     futility=20 if r['robust']['p_kind']=='conservative_futility' else None)
        assert q['p']==r['p'] and q['statistics']==r['robust']['statistics']
        tests.append(dict(graph=g['id'],model=r['model'],p=q['p'],reject=r['reject'],reproduced=True))
    events=read(OUT/'elementary-events.json');lookup={(r['graph'],r['model'],r['event']):r for r in events}
    for r in allrows:
        for label in ('Luby',r['model']):
            a=bank(label,r['graph']);X=a['colors'][:500][a['status'][:500]==0];v=measurements(X,gs[r['graph']])
            for event in EVENTS:
                z=lookup[(r['graph'],label,event)];assert z['count']==int(v[event].sum()) and z['total']==len(X)
        if 'witness' in r:
            for label,key in [('Luby','Luby'),(r['model'],'model')]:
                a=bank(label,r['graph']);X=a['colors'][500:][a['status'][500:]==0]
                count=int(event_values(r['witness']['event'],X,gs[r['graph']]).sum())
                assert count==r['witness']['count_'+key] and len(X)==r['witness']['n_'+key]
    fc=read(OUT/'factorial-contrasts.json');assert len(fc)==f['factorial_contrasts']
    assert not any(r.get('bounds_conflict',False) for r in fc)
    budget=read(ROOT/'budget.json');assert budget['seconds']<=budget['limit']==43200. and budget['active_wall'] is None
    pd.DataFrame(manifests).to_csv(OUT/'sampling-manifest.csv',index=False)
    result=dict(passed=True,updated=now(),protocol_commit=commit,graphs=graph_audit,
                banks=len(manifests),new_algorithm_seeds=len(new),distinct_old_seeds=len(np.unique(old)),old_banks_checked=oldbank_count,
                distinct_transfer_test_seeds=len(auxiliary),overlap_seeds=0,statuses_main_and_pilots=allstatus.tolist(),
                mechanism_forbidden_candidates=int(mechanism_errors[0]),mechanism_unsafe_commits=int(mechanism_errors[1]),
                teacher_bank_replay=True,test_replays=tests,factorial_contrasts=len(fc),gpu_process_hours=budget['seconds']/3600,
                source_hashes=f['source_hashes'])
    atomic_json(OUT/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','test_replays')}),flush=True)

if __name__=='__main__':main()
