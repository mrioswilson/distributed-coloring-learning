"""Full artifact audit: cardinalities, paired traces, failures, laws and LRT replay."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from dcl.study.storage import atomic_json, file_hash
from .cohort import palette_for
from .config import ROOT, RESULTS, PROTOCOL, TREATMENTS
from .report import write_csv
from .statistics import law_metrics, likelihood_curve


def run():
    start=time.monotonic();checks={}
    frozen=json.loads((RESULTS/'preregistration-environment.json').read_text())
    protocol_bytes=subprocess.check_output(['git','show',f'f516530:{PROTOCOL}'])
    assert hashlib.sha256(protocol_bytes).hexdigest()==file_hash(PROTOCOL)==frozen['protocol_sha256']
    checks['protocol_commit_and_bytes_unchanged']=True
    for p in ('src/dcl/universal/model.py','src/dcl/paperstudy/models.py','src/dcl/paperstudy/training.py'):
        assert file_hash(p)==frozen['environment']['source_hashes'][p]
    checks['historical_model_and_training_sources_unchanged']=True
    main=json.loads((RESULTS/'frozen-checkpoints.json').read_text())
    historical=json.loads((RESULTS/'historical-e1-checkpoints.json').read_text())
    assert len(main)==30 and len(historical)==5
    assert Counter(s['treatment'] for s in main)=={t:5 for t in TREATMENTS}
    assert all(s['epoch']==120 and file_hash(s['checkpoint'])==s['sha256'] for s in main+historical)
    checks['35_frozen_checkpoint_hashes']=True
    assert not list(ROOT.rglob('*.pt'))
    checks['no_new_research_checkpoints']=True
    graphs=json.loads((RESULTS/'graph-manifest.json').read_text());assert len(graphs)==76
    graphmap={g['id']:g for g in graphs};arrays={}
    for g in graphs:
        x=np.load(g['file']);e=x['edges'];d=x['degree']
        assert hashlib.sha256(e.tobytes()).hexdigest()==g['edges_sha256']
        assert np.array_equal(np.bincount(e[1],minlength=g['n']),d)
        assert not np.any(e[0]==e[1])
        assert set(map(tuple,e.T))==set(map(tuple,e[::-1].T))
        arrays[g['id']]=(e,d)
    checks['graph_arrays_and_degrees']=True
    primary={s['label'] for s in main}|{'exact-teacher'}
    seen=set();statuses=Counter();traces={};failed=0;observer=Counter()
    paths=sorted((ROOT/'graphs').rglob('*.json'))
    assert len(paths)==18848
    for path in paths:
        row=json.loads(path.read_text());g=row['graph'];ident=(row['model'],g['id'],row['regime'],row['tape'])
        assert ident not in seen and row['model'] in primary;seen.add(ident)
        assert row['protocol_sha256']==frozen['protocol_sha256'] and g==graphmap[g['id']]
        pending=g['n']
        for step in row['trace']:
            assert step['pending']==pending and 0<=step['commits']<=step['participations']<=pending
            pending-=step['commits']
        assert pending==row['pending'] and len(row['trace'])==row['rounds']<=256
        assert row['out_of_list']==0
        if row['status']=='success':assert pending==0 and row['conflicts']==0
        else:
            x=np.load(path.with_suffix('.npz'));colors=x['colors'];pal=x['palette'];e,d=arrays[g['id']]
            assert np.array_equal(pal,palette_for(row['regime'],d))
            assert np.all((colors>=-1)&(colors<pal)) and np.sum(colors<0)==pending
            conflicts=int(np.sum((colors[e[0]]>=0)&(colors[e[0]]==colors[e[1]]))//2)
            assert conflicts==row['conflicts'];failed+=1
            if row['status']=='progress_cutoff':assert row['rounds']==256 and pending>0
        statuses[row['status']]+=1
        if 'exact_eligible_commits' in row:
            assert row['false_rejects']==row['exact_eligible_commits']-sum(s['commits']-s['unsafe_commits'] for s in row['trace'])
            observer[row['treatment']]+=1
        if g['family']=='REGULAR':
            traces[ident]=[{k:v for k,v in s.items() if k not in ('exact_eligible_commits','false_rejects')} for s in row['trace']]
    assert Counter(i[0] for i in seen)=={m:608 for m in primary}
    regular_pairs=0
    for (m,g,regime,tape),trace in traces.items():
        if regime=='GLOBAL':
            assert traces[(m,g,'LOCAL',tape)]==trace;regular_pairs+=1
    checks['18848_complete_unique_paired_cases']=True
    checks['all_failed_final_colorings_independently_rechecked']=failed
    checks['regular_global_local_trace_pairs_identical']=regular_pairs
    states=(json.loads((RESULTS/'state-manifest.json').read_text())+
            json.loads((RESULTS/'autonomous-state-manifest.json').read_text())+
            json.loads((RESULTS/'historical-state-manifest.json').read_text()))
    assert len(states)==93 and len({s['id'] for s in states})==93
    state_map={s['id']:s for s in states}
    for s in states:
        assert s['colors'][s.get('center',0)]==-1
        assert np.all(np.array(s['colors'])<np.array(s['palettes']))
    mc_seeds=set();lawseen=set();b3_replays=0;candidate_counts=[]
    for path in sorted((ROOT/'distributions').rglob('*.json')):
        record=json.loads(path.read_text());r=record['metrics'];s=state_map[r['state']]
        assert (r['model'],r['state']) not in lawseen;lawseen.add((r['model'],r['state']))
        x=np.load(path.with_suffix('.npz'));a=x['available'];counts=x['estimation_counts'];test=x['test_counts'];k=s['k']
        assert len(counts)==len(test)==k and counts.sum()==32768 and test.sum()==8192
        assert np.array_equal(a,s['available']) and len(x['quadrature_choices'])==2049
        assert np.all((x['boundary_rc']>=0)&(x['boundary_rc']<1))
        metric=law_metrics(counts,a)
        for key in ('tv','tv_lower','tv_upper','js','hellinger','entropy','effective_collision','effective_entropy','max_mass','forbidden_mass'):
            assert np.isclose(metric[key],r[key],atol=1e-12,rtol=1e-12),(r['model'],s['id'],key)
        for value in r['seeds'].values():assert value not in mc_seeds;mc_seeds.add(value)
        forbidden=np.ones(k,bool);forbidden[a]=False
        summary=dict(model=r['model'],state=s['id'],k=k,estimation_forbidden=int(counts[forbidden].sum()),
                     independent_test_forbidden=int(test[forbidden].sum()),
                     deterministic_quadrature_forbidden=int(np.sum(~np.isin(x['quadrature_choices'],a))),
                     deterministic_boundary_forbidden=int(np.sum(~np.isin(x['boundary_choices'],a))),
                     effective_entropy=r['effective_entropy'],effective_collision=r['effective_collision'])
        if s.get('b3'):
            trial=np.load(path.parent/f'{s["id"]}-b3.npz');model=trial['model'];teacher=trial['teacher']
            assert model.shape==teacher.shape==(256,2000)
            assert np.all((model>=0)&(model<k)) and np.all(np.isin(teacher,a))
            summary['classifier_test_forbidden']=int(np.sum(~np.isin(model,a)))
            replay=likelihood_curve(counts,a,model,teacher,np.random.default_rng(r['seeds']['calibration']),
                                    tie_rng=np.random.default_rng(r['seeds']['ties']))
            for actual,saved in zip(replay,record['b3']):
                for key,value in actual.items():assert np.allclose(value,saved[key],atol=1e-12,rtol=1e-12),(r['model'],s['id'],key)
            b3_replays+=1
        else:assert not record['b3']
        candidate_counts.append(summary)
    assert len(lawseen)==3255 and b3_replays==210 and len(mc_seeds)==3255*6
    assert Counter(m for m,s in lawseen)=={m['label']:93 for m in main+historical}
    checks['all_3255_law_estimates_recomputed']=True
    checks['all_19530_mc_split_seeds_distinct']=True
    checks['all_1260_classifier_rows_replayed_from_independent_banks']=True
    checks['observer_coverage_by_treatment']=dict(observer)
    invariant=json.loads((RESULTS/'adapter-verification.json').read_text())
    assert invariant['passed'] and invariant['trained_networks']==30 and invariant['legal_witnesses_correct']
    assert len(invariant['local_message_witnesses'])==1280
    checks['trained_network_invariances_and_local_witnesses']=True
    untracked={}
    for previous in ('/workspaces/AlgoLearning/distributed-coloring-learning','/workspaces/AlgoLearning/ucgnn-paper-study','/workspaces/AlgoLearning/ucgnn-extreme-degree-study'):
        status=subprocess.check_output(['git','-C',previous,'status','--porcelain'],text=True).splitlines()
        assert all(line.startswith('?? ') for line in status),(previous,status)
        if status:untracked[previous]=status
    checks['historical_tracked_worktrees_unchanged']=True
    checks['untracked_historical_files_left_untouched']=untracked
    write_csv(RESULTS/'candidate-sample-audit.csv',candidate_counts)
    epoch=int(subprocess.check_output(['git','show','-s','--format=%ct','f516530'],text=True))
    first=min(p.stat().st_mtime for p in paths);assert first>=epoch
    result=dict(passed=True,checks=checks,status_counts=dict(statuses),runtime_seconds=time.monotonic()-start,
                protocol_commit='f516530',first_graph_result_utc=datetime.fromtimestamp(first,timezone.utc).isoformat(),
                protocol_commit_utc=datetime.fromtimestamp(epoch,timezone.utc).isoformat(),
                note='This audits measured coverage and recorded data; it does not prove universal neural termination.')
    atomic_json(RESULTS/'completion-audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
