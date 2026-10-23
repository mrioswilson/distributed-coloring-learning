import json,itertools
from pathlib import Path
import numpy as np
from .core import *
from .analyze import read,bank
from dcl.study.storage import provenance

def main():
    freeze=read(OUT/'freeze.json');queue=read(ROOT/'queue.json');gs={g['id']:g for g in read(OUT/'graphs.json')}
    assert freeze['protocol_sha256']==file_hash('docs/GLOBAL_COLORING_DISTRIBUTION_PROTOCOL.md')
    assert freeze['graphs_sha256']==file_hash(OUT/'graphs.json')
    assert freeze['checkpoints_sha256']==file_hash(OUT/'checkpoints.json')
    assert len(specs())==15 and all(file_hash(s['checkpoint'])==s['sha256'] for s in read(OUT/'checkpoints.json'))
    from dcl.paperstudy.training import load_model
    import pandas as pd
    historical=read('/workspaces/AlgoLearning/ucgnn-local-palette-study/results/ucgnn-local-palettes/historical-e1-checkpoints.json')
    old_hashes={s['sha256'] for s in historical};identity=[]
    for s in read(OUT/'checkpoints.json'):
        assert s['sha256'] not in old_hashes
        model,ck=load_model(s['checkpoint'],'cpu')
        assert ck['epoch']==120 and ck['complete']
        assert ck['paper_config']['seed']==s['seed'] and ck['paper_config']['treatment']==s['treatment']
        identity.append(dict(label=s['label'],seed=s['seed'],epoch=ck['epoch'],parameters=ck['parameters'],sha256=s['sha256'],checkpoint=s['checkpoint'],training_sources_sha256=ck['training_sources_sha256'],historical_e1_overlap=False))
        del model,ck
    pd.DataFrame(identity).to_csv(OUT/'checkpoint-content-audit.csv',index=False)
    seen=set();rows=[];status=np.zeros(4,dtype=int)
    for task in queue['tasks']:
        assert task['status']=='complete',task['id']
        assert file_hash(task['file'])==task['sha256']
        a=np.load(task['file']);g=gs[task['graph']];e=np.array(g['edges']);degree=np.array(g['degrees']);n=g['n']
        assert a['colors'].shape==(task['m'],n)
        label='Luby' if task['model'] is None else task['model']['label']
        expected=[seed(task['stage'],g['id'],label,'run',i) for i in range(task['m'])]
        assert np.array_equal(a['seeds'],expected)
        seeds=set(map(int,a['seeds']));assert len(seeds)==task['m'] and not(seeds&seen);seen|=seeds
        conflict=((a['colors'][:,e[0]]==a['colors'][:,e[1]])&(a['colors'][:,e[0]]>=0)).sum(1)//2
        outside=((a['colors']>=degree[None,:]+1)|(a['colors']< -1)).sum(1)
        complete=(a['colors']>=0).all(1)
        valid=complete&(conflict==0)&(outside==0)
        assert np.array_equal(valid,a['status']==0)
        assert a['rounds'].max()<=128
        h=hashlib.sha256(np.int64(n).tobytes()+e.astype(np.int64).tobytes()).hexdigest();assert h==g['sha256']
        actual=np.bincount(e[0],minlength=n);assert np.array_equal(actual,degree)
        status+=np.bincount(a['status'],minlength=4)
        rows.append(dict(task=task['id'],sha256=task['sha256'],runs=task['m'],max_rounds=int(a['rounds'].max()),
                         edge_conflicts=int(conflict.sum()),outside_list=int(outside.sum()),
                         first_rng_initial_state=json.loads(str(a['first_rng_initial_state'])),
                         final_rng_states_sha256=str(a['final_rng_states_sha256'])))
    localized=read(OUT/'localized-positive-control.json')
    for item in localized:assert file_hash(item['file'])==item['sha256']
    primary=read(OUT/'primary.json');assert len(primary)==795
    assert len({(r['graph'],r['model']) for r in primary})==795
    assert all(r['B']==19999 and r['gamma']==1 and r['planned_m']==250 for r in primary)
    assert queue['gpu_seconds']<=43200
    # Replay fixed, prospectively selected permutation cases against raw arrays.
    replay=[]
    for s in [s for s in specs() if s['seed']==41000]:
        g='path3';a=bank('confirmatory','Luby',g);b=bank('confirmatory',s['label'],g)
        r=mmd_test(a['colors'][a['status']==0],b['colors'][b['status']==0],np.random.default_rng(seed('primary-permutation',g,s['label'])),B=19999)
        archived=next(x for x in primary if x['graph']==g and x['model']==s['label'])
        assert r['p']==archived['p'] and abs(r['mmd2_unbiased']-archived['mmd2_unbiased'])<1e-12
        replay.append(dict(graph=g,model=s['label'],p=r['p']))
    atomic_json(OUT/'audit.json',dict(passed=True,banks=len(rows),independent_run_seeds=len(seen),statuses=status.tolist(),
         gpu_hours=queue['gpu_seconds']/3600,bank_validation=rows,permutation_replay=replay,environment=provenance(),
         limitations=['No universal safety/progress proof for neural policies','No graph-population inference',
                      'Power intervals conditional on empirical banks','Pointwise effect intervals are conservative']))
    print('Audit passed',len(rows),'banks',len(seen),'independent run seeds')
if __name__=='__main__':main()
