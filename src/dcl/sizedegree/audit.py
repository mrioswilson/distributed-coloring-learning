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
from .capacity import bound as capacity_bound
from dcl.globaldist.core import seed as run_seed,holm
from dcl.globaldist.sampling import sample
from dcl.robustdist.stats import joint_test
from dcl.robustdist.confirmation import event_values
from dcl.universal.rules import exact_commit_labels,available_matrix

def main():
    f=verify_frozen();record=read(OUT/'protocol-commit.json');protocol_commit=record['commit']
    frozen_doc=subprocess.check_output(['git','show',protocol_commit+':docs/FACTORIAL_SIZE_DEGREE_PROTOCOL.md'])
    import hashlib
    assert hashlib.sha256(frozen_doc).hexdigest()==f['protocol_sha256']
    prereg=datetime.fromisoformat(subprocess.check_output(['git','show','-s','--format=%cI',protocol_commit],text=True).strip())
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
            if capacity_bound(g['degrees'])['A_deterministically_true']:
                assert all(len(set(row))<=g['delta'] for row in x[status==0])
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
        assert t['source_sha256']==file_hash('src/dcl/sizedegree/mechanism.py')
        with np.load(path) as a:
            newseeds.append(a['seeds']);mechanism_errors+=a['observer_errors'].sum((0,1))
            label='Luby' if t['model'] is None else t['model']['label'];g=gs[t['graph']]
            expected=[run_seed('factorial-v1/mechanism',g['id'],label,'run',i) for i in range(8)]
            assert np.array_equal(a['seeds'],np.array(expected,np.uint64))
            assert (a['commit_counts'].sum((0,2))==(a['colors']>=0).sum(1)).all()
            edges=np.concatenate([np.array(g['edges'])+i*g['n'] for i in range(8)],axis=1)
            pal=np.tile(np.array(g['degrees'])+1,8)
            for j,S in enumerate(a['states']):
                candidate=a['node_candidates'][j];part=a['node_participate'][j];commit=a['node_commit'][j]
                after=np.where(commit,candidate,S)
                expected=a['states'][j+1] if j+1<len(a['states']) else a['colors']
                assert np.array_equal(after,expected)
                safe=exact_commit_labels(S.ravel(),edges,part.ravel(),candidate.ravel()).reshape(S.shape)
                assert not (commit&~safe).any()
                available=available_matrix(S.ravel(),edges,pal).sum(1).reshape(S.shape)
                assert np.array_equal(available,a['node_available_counts'][j])
        manifests.append(dict(phase='mechanism',id=t['id'],graph=t['graph'],model=label,file=str(path),sha256=t['sha256'],attempts=8,seconds=t['seconds']))
    assert mechanism_errors[1]==0
    intervention_streams=0;intervention_unsafe=0
    ipath=ROOT/'candidate-intervention/queue.json'
    if ipath.exists():
        iq=read(ipath)
        for prefix,module,document in [
            ('protocol','intervention','FACTORIAL_CANDIDATE_INTERVENTION_PLAN.md'),
            ('witness-addendum','intervention_witnesses','FACTORIAL_INTERVENTION_WITNESS_ADDENDUM.md')]:
            record_name='protocol-commit.json' if prefix=='protocol' else prefix+'-commit.json'
            plan_name='plan.json' if prefix=='protocol' else prefix+'.json'
            record=read(OUT/'candidate-intervention'/record_name);plan=read(OUT/'candidate-intervention'/plan_name)
            assert record['preexisting_intervention_banks']==0
            source='src/dcl/sizedegree/'+module+'.py';doc='docs/'+document
            assert file_hash(source)==plan['source_sha256']
            assert file_hash(doc)==plan['plan_sha256' if prefix=='protocol' else 'document_sha256']
            assert subprocess.check_output(['git','show',record['commit']+':'+source])==Path(source).read_bytes()
            assert subprocess.check_output(['git','show',record['commit']+':'+doc])==Path(doc).read_bytes()
            committed=datetime.fromisoformat(subprocess.check_output(['git','show','-s','--format=%cI',record['commit']],text=True).strip())
            assert datetime.fromisoformat(iq['created'])>committed
        resource_path=OUT/'candidate-intervention/resource-freeze.json'
        resource=read(resource_path) if resource_path.exists() else None
        for t in iq.get('pilot_tasks',[])+iq.get('tasks',[]):
            if t['status']!='complete':continue
            if t['phase']=='confirm':
                assert resource is not None
                assert t['m']==resource['pairs_per_cell']
                assert datetime.fromisoformat(t['completed'])>datetime.fromisoformat(resource['created'])
            path=Path(t['file']);assert file_hash(path)==t['sha256']
            g=gs[t['graph']];label=t['model']['label']
            with np.load(path) as c:
                expected=np.array([run_seed('factorial-v1/candidate-intervention-'+t['phase'],g['id'],label,'run',i) for i in range(t['m'])],np.uint64)
                assert np.array_equal(c['seeds'],expected)
                newseeds.append(c['seeds']);intervention_streams+=len(c['seeds']);intervention_unsafe+=int(c['unsafe_commits'].sum())
                for side in ('Luby','intervened'):
                    X=c['colors_'+side];st=c['status_'+side];ee=np.array(g['edges'])
                    valid=(X>=0).all(1)&(X<=np.array(g['degrees'])).all(1)&(X[:,ee[0]]!=X[:,ee[1]]).all(1)
                    assert valid[st==0].all() and not valid[st==1].any()
                    assert (X[st==2]<0).any(1).all() and (c['rounds_'+side][st==2]==128).all()
                same=(c['status_Luby']==c['status_intervened'])&((c['status_Luby']!=0)|(c['colors_Luby']==c['colors_intervened']).all(1))
                assert np.array_equal(~same,c['coarsened_mismatch'])
            manifests.append(dict(phase='candidate-intervention-'+t['phase'],id=t['id'],graph=g['id'],model=label,
                                  file=str(path),sha256=t['sha256'],attempts=2*t['m'],pairs=t['m'],seconds=t['seconds']))
    new=np.concatenate(newseeds);assert len(np.unique(new))==len(new)
    oldseeds=[];oldbank_count=0
    for root,sub in [(GLOBAL,'ucgnn-global-distribution'),(BASE,'ucgnn-robust-distribution')]:
        if sub=='ucgnn-global-distribution':
            reference=read(root/'results'/sub/'sampling-manifest.json')['tasks']
            expected={Path(t['file']).name:t['sha256'] for t in reference}
        else:
            reference=pd.read_csv(root/'results'/sub/'sampling-manifest.csv')
            expected={Path(r.file).name:r.sha256 for r in reference.itertuples()}
        for path in (root/'data'/sub/'banks').glob('*.npz'):
            assert path.name in expected and file_hash(path)==expected[path.name]
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
    result=dict(passed=True,updated=now(),protocol_commit=protocol_commit,graphs=graph_audit,
                banks=len(manifests),new_algorithm_seeds=len(new),distinct_old_seeds=len(np.unique(old)),old_banks_checked=oldbank_count,
                distinct_transfer_test_seeds=len(auxiliary),overlap_seeds=0,statuses_main_and_pilots=allstatus.tolist(),
                mechanism_forbidden_candidates=int(mechanism_errors[0]),mechanism_unsafe_commits=int(mechanism_errors[1]),
                candidate_intervention_pair_streams=intervention_streams,candidate_intervention_unsafe_commits=intervention_unsafe,
                teacher_bank_replay=True,test_replays=tests,factorial_contrasts=len(fc),gpu_process_hours=budget['seconds']/3600,
                source_hashes=f['source_hashes'])
    atomic_json(OUT/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','test_replays')}),flush=True)

if __name__=='__main__':main()
