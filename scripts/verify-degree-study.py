#!/usr/bin/env python3
"""Audit every reported row, checkpoint and preserved historical artifact."""
from collections import Counter
import argparse
import json
from pathlib import Path
import time
import numpy as np
from dcl.degree.inventory import ROOT
from dcl.degree.run import source_hashes
from dcl.degree.graphs import graph_for
from dcl.study.storage import atomic_json,file_hash


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--portable',action='store_true',help='Verify bundled data without requiring external original worktrees')
    args=parser.parse_args()
    tick=time.perf_counter()
    inventory=json.loads((ROOT/'checkpoint-inventory.json').read_text())
    extra=json.loads((ROOT/'supplemental-checkpoint-inventory.json').read_text())
    assert len(inventory)==195 and len(extra)==30
    assert sum(r['primary'] for r in inventory)==105
    weights=inventory+extra
    for r in weights:
        assert file_hash(r['path'])==r['sha256']
        if not args.portable:assert file_hash(r['source_absolute'])==r['sha256']
    assert source_hashes()==json.loads((ROOT/'confirmatory/design.json').read_text())['sources']
    old=json.loads((ROOT/'historical-preservation.json').read_text())
    if not args.portable:
        for path,sha in old['files'].items():assert file_hash(path)==sha,path
    graph_cache={}
    totals={}
    cases=0
    expected_counts={'pilot':3570,'confirmatory':810000,'larger-size-pilot':240}
    for stage in ('pilot','confirmatory','larger-size-pilot'):
        summary=json.loads((ROOT/stage/'index.json').read_text())
        executions=successes=0
        signatures=set()
        for r in summary:
            key=r['checkpoint_id'],r['condition']['id']
            assert key not in signatures
            signatures.add(key)
            assert file_hash(r['rows_path'])==r['rows_sha256']
            rows=json.loads(Path(r['rows_path']).read_text())
            assert len(rows)==r['executions'] and sum(x['success'] for x in rows)==r['successes']
            assert len({x['case_id'] for x in rows})==len(rows)
            for row in rows:
                c=r['condition']
                assert row['n']==c['n'] and row['delta']==row['actual_delta']==c['delta']
                assert row['family'] in c['families']
                gkey=row['family'],row['n'],row['delta'],row['graph_seed']
                if gkey not in graph_cache:
                    g=graph_for(*gkey)
                    degree=np.array([g.degree(v) for v in range(len(g))])
                    edges=np.array(list(g.edges()),dtype=int).reshape(-1,2).T
                    graph_cache[gkey]=(degree,edges)
                degree,edges=graph_cache[gkey]
                colors=np.array(row['colors'])
                assert len(colors)==row['n']
                palette=degree+1 if row['palette']=='local' else np.full(row['n'],row['delta']+1)
                a,b=edges
                conflicts=int(np.sum((colors[a]>=0)&(colors[a]==colors[b])))
                violations=int(np.sum((colors< -1)|(colors>=palette)))
                incomplete=int(np.sum(colors<0))
                assert (conflicts,violations,incomplete)==(row['conflicts'],row['palette_violations'],row['incomplete'])
                passed=not conflicts and not violations and not incomplete and row['invariants'] and row['failure'] is None
                assert bool(passed)==row['success']
                cases+=1
            executions+=r['executions'];successes+=r['successes']
        assert executions==expected_counts[stage]
        exact=json.loads((ROOT/stage/'exact-validation.json').read_text())
        exact_count=0
        for meta in exact['files']:
            assert file_hash(meta['path'])==meta['sha256']
            rows=json.loads(Path(meta['path']).read_text())
            assert len(rows)==meta['executions'] and all(r['success'] for r in rows)
            exact_count+=len(rows)
        totals[stage]=dict(executions=executions,successes=successes,exact_references=exact_count,
                          checkpoint_conditions=len(summary))
    replays=json.loads((ROOT/'failure-replays/index.json').read_text())
    final_index=json.loads((ROOT/'confirmatory/index.json').read_text())
    assert len(replays)==sum(r['successes']<r['executions'] for r in final_index)
    for path in replays:assert json.loads(Path(path).read_text())['replay_matches_original']
    required=['MULTITEACHER_DEGREE_EXTENSIBILITY','FROZEN_CHECKPOINT_COMPATIBILITY','DEGREE_GENERALIZATION_PROTOCOL',
              'DEGREE_GENERALIZATION_RESULTS','DEGREE_GENERALIZATION_FAILURE_ANALYSIS','MULTITEACHER_PAPER_CLOSURE_REVIEW',
              'HIGHER_DEGREE_TRAINING_PROPOSAL']
    for name in required:assert (Path('docs')/(name+'.md')).is_file()
    closure=Path('docs/MULTITEACHER_PAPER_CLOSURE_REVIEW.md').read_text()
    for i in range(1,16):assert f'**P{i}.**' in closure
    pdf=json.loads((ROOT/'paper-build.json').read_text())
    assert file_hash('paper/ucgnn/main.pdf')==pdf['pdf_sha256'] and pdf['compiled'] and not pdf['unresolved_diagnostics']
    report=dict(complete=True,verified_student_rows=cases,verified_checkpoints=len(weights),
                preserved_historical_files=0 if args.portable else len(old['files']),external_originals_checked=not args.portable,
                stages=totals,verified_failure_replays=len(replays),
                scientific_training_updates=0,weights_unchanged=True,pdf_compiled=True,
                wall_seconds=time.perf_counter()-tick,
                test_scope='Historical regression tests contain disposable native-domain optimization smokes; no resulting weights enter this study.')
    atomic_json(ROOT/('portable-verification.json' if args.portable else 'verification.json'),report)
    print(json.dumps(report))
