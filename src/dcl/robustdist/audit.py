"""Final read-only scientific checks; the audit artifact is the only output."""
import hashlib,json,subprocess,time
from fractions import Fraction
from datetime import datetime
import numpy as np
import pandas as pd
from .common import *
from .stats import holm,joint_test,multiscale_test
from .confirmation import pending_refinements
from .controls_v2 import draw
from dcl.globaldist.core import seed as run_seed
from dcl.study.storage import provenance

def main():
    start=time.monotonic();f=read(OUT/'freeze.json');receipt=read(OUT/'protocol-commit.json')
    assert f['protocol_sha256']==file_hash('docs/ROBUST_GLOBAL_DISTRIBUTION_PROTOCOL.md')==receipt['protocol_sha256']
    raw=subprocess.check_output(['git','show',receipt['commit']+':docs/ROBUST_GLOBAL_DISTRIBUTION_PROTOCOL.md'])
    assert hashlib.sha256(raw).hexdigest()==f['protocol_sha256'] and receipt['before_confirmation']
    for p,h in f['source_hashes'].items():assert file_hash(p)==h,p
    assert read(OUT/'instrument-acceptance.json')['passed']
    assert file_hash(OUT/'graphs.json')==f['graphs_sha256']
    assert file_hash(OUT/'checkpoints.json')==f['checkpoints_sha256']
    gs={g['id']:g for g in graphs()};assert len(gs)==53 and len({g['sha256'] for g in gs.values()})==51
    assert gs['bipartite-n256-d3']['delta']==4
    for g in gs.values():
        e=np.array(g['edges'],np.int64).reshape(2,-1);d=np.bincount(e[0],minlength=g['n'])
        assert np.array_equal(d,g['degrees']) and int(d.max())==g['delta']
        assert hashlib.sha256(np.int64(g['n']).tobytes()+e.tobytes()).hexdigest()==g['sha256']
    identities=[]
    from dcl.paperstudy.training import load_model
    old_e1=read('/workspaces/AlgoLearning/ucgnn-local-palette-study/results/ucgnn-local-palettes/historical-e1-checkpoints.json')
    historical_hashes={s['sha256'] for s in old_e1}
    assert len(specs())==15
    for s in specs():
        assert s['treatment'] in ('normalized','deepsets','transformer') and s['sha256'] not in historical_hashes
        assert file_hash(s['checkpoint'])==s['sha256']
        model,ck=load_model(s['checkpoint'],'cpu')
        assert not model.training and ck['complete'] and ck['epoch']==120
        assert ck['paper_config']['seed']==s['seed'] and ck['paper_config']['treatment']==s['treatment']
        identities.append(dict(label=s['label'],epoch=120,seed=s['seed'],sha256=s['sha256'],historical_e1=False))
        del model,ck
    seen=set();oldseen=set()
    for t in read(OLD/'data/ucgnn-global-distribution/queue.json')['tasks']:
        a=np.load(OLD/t['file']);oldseen.update(map(int,a['seeds']))
    phases=[];banks=[];total=np.zeros(4,np.int64)
    for qfile in sorted(DATA.glob('queue-*.json')):
        q=read(qfile);counts=np.zeros(4,np.int64)
        if q['phase']=='confirmatory':
            assert len(q['tasks'])==848 and all(t['status']=='complete' for t in q['tasks'])
            assert datetime.fromisoformat(q['created'])>=datetime.fromisoformat(receipt['created'])
        for t in q['tasks']:
            assert t['status']=='complete' and file_hash(t['file'])==t['sha256'],t['id']
            a=np.load(t['file']);g=gs[t['graph']];x=a['colors'];st=a['status'];e=np.array(g['edges'],int).reshape(2,-1)
            label='Luby' if t['model'] is None else t['model']['label']
            expected=[run_seed(t['stage'],g['id'],label,'run',i) for i in range(t['m'])]
            assert np.array_equal(a['seeds'],expected)
            ss=set(map(int,a['seeds']));assert len(ss)==t['m'] and not ss&seen and not ss&oldseen
            seen|=ss;assert x.shape==(t['m'],g['n']) and np.isin(st,[0,1,2,3]).all()
            for lo in range(0,len(x),128):
                v=x[lo:lo+128];complete=(v>=0).all(1)
                ok=complete&(v<=np.array(g['degrees'])).all(1)&(v[:,e[0]]!=v[:,e[1]]).all(1)
                s=st[lo:lo+128];assert np.array_equal(ok,s==0)
                assert (~complete[s==2]).all() and (complete[s==1]).all()
            assert int(a['rounds'].max())<=128
            counts+=np.bincount(st,minlength=4)
            banks.append(dict(phase=q['phase'],id=t['id'],file=t['file'],sha256=t['sha256'],runs=t['m'],statuses=np.bincount(st,minlength=4).tolist(),
                              rng_initial=json.loads(str(a['first_rng_initial_state'])),rng_final_hash=str(a['final_rng_states_sha256'])))
        total+=counts;phases.append(dict(phase=q['phase'],banks=len(q['tasks']),statuses=counts.tolist()))
    budget=read(DATA/'budget.json');assert budget['seconds']<=43200 and budget['active_wall'] is None
    primary=read(OUT/'primary.json');secondary=read(OUT/'secondary.json')
    assert len(primary)==795 and len(secondary)==405
    assert len({(r['graph'],r['model']) for r in primary})==795
    assert all(r['total_per_generator']==500 for r in primary)
    for rows in (primary,secondary):
        assert np.allclose(holm([r['p'] for r in rows]),[r['p_holm'] for r in rows],atol=0,rtol=0)
        assert np.allclose(holm([r['old_p'] for r in rows]),[r['old_p_holm'] for r in rows],atol=0,rtol=0)
        assert not pending_refinements(rows,'p','p_lower_bound') and not pending_refinements(rows,'old_p','old_p_lower_bound')
        for r in rows:
            assert r['graph_sha256']==gs[r['graph']]['sha256']
            assert r['p']>0 and r['old_p']>0 and r['robust']['B']==r['old']['B']==19999
            assert r['robust']['selection_per_class'][0]+r['robust']['evaluation_per_class'][0]==r['actual_m_Luby']
    witnesses=read(OUT/'explanatory-witnesses.json');assert len(witnesses)==795
    assert np.allclose(holm([r['p'] for r in witnesses]),[r['p_holm'] for r in witnesses],atol=0,rtol=0)
    assert all(r['n_Luby']<=250 and r['n_model']<=250 for r in witnesses)
    small=read(OUT/'small-reference.json');assert len(small)==105
    for r in small:
        assert sum(Fraction(v) for v in r['exact_law'].values())==1
        assert r['approximate_equivalence']==(r['simultaneous_tv_ci'][1]<=f['tv_tolerance'])
    # Every synthetic array, not only its rejection count, is archived and valid.
    synthetic=0;classifier_underflows=[];benchseeds=set()
    for file in sorted((OUT/'benchmark-trials').glob('*.json')):
        r=read(file);assert file_hash(r['raw_file'])==r['raw_sha256']
        a=np.load(r['raw_file']);_,g,_=draw(r['case'],1,rng('audit-graph-only',r['id']),D=r['D'],effect=r['effect'])
        assert valid(a['X'],g).all() and valid(a['Y'],g).all()
        for side in ('A','B'):
            s=seed(r['stage'],r['case'],r['m'],r['trial'],r['D'],r['effect'],side)
            assert s not in benchseeds;benchseeds.add(s)
        for method,p in r['p_values'].items():
            if method=='classifier' and p==0:classifier_underflows.append(r['id'])
            else:assert 0<p<=1,(r['id'],method,p)
        synthetic+=1
    replay=[]
    for gid,label in [('path3','normalized-s41000'),('star-d200','transformer-s41000')]:
        row=next(r for r in primary if r['graph']==gid and r['model']==label);g=gs[gid]
        a=np.load(DATA/'banks'/f'confirmatory__Luby__{gid}.npz');b=np.load(DATA/'banks'/f'confirmatory__{label}__{gid}.npz')
        X=a['colors'][:500][a['status'][:500]==0];Y=b['colors'][:500][b['status'][:500]==0]
        result=joint_test(X,Y,g,rng('confirmatory',gid,label,500,'omnibus'),B=19999,futility=None if row.get('robust_refined_full_B') else 20)
        assert result['p']==row['p']
        assert abs(result['statistics']['omnibus']-row['statistic'])<1e-9
        replay.append(dict(graph=gid,model=label,p=result['p']))
    audit=dict(passed=True,protocol_commit=receipt['commit'],banks=len(banks),independent_algorithm_run_seeds=len(seen),no_overlap_with_old_run_seeds=len(oldseen),
         statuses=total.tolist(),phase_summaries=phases,gpu_process_hours=budget['seconds']/3600,synthetic_trials_verified=synthetic,
         independent_synthetic_generator_seeds=len(benchseeds),primary_hypotheses=795,primary_rejections=sum(r['reject'] for r in primary),
         secondary_hypotheses=405,simultaneous_witness_cells=795,small_reference_cells=105,checkpoints=identities,
         bank_validation=banks,replays=replay,seconds=time.monotonic()-start,
         numerical_exact_classifier_underflows=dict(count=len(classifier_underflows),trials=classifier_underflows,
             note='These are exact hypergeometric-tail float underflows on extreme benchmark accuracy, not Monte Carlo p=0. Their alpha=.05 decisions are unchanged. Main n=250-per-class classifier tails are representable; no zero Monte Carlo p-value is allowed.'),
         limitations=['No universal finite-sample power','Unknown relocated higher-order interactions can remain undetected','Rare events may be insufficiently sampled','Ideal clique theorem is not a bitwise hardware certificate','Fixed graph/training cohort, no population inference'],environment=provenance())
    atomic_json(OUT/'audit.json',audit)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in banks]).to_csv(OUT/'sampling-manifest.csv',index=False)
    print('AUDIT PASSED',len(banks),'banks',len(seen),'distinct new run seeds',flush=True)
if __name__=='__main__':main()
