#!/usr/bin/env python3
"""Fresh deterministic replay of 32 frozen cases for each primary core/teacher."""
import json
from pathlib import Path
import time
import torch
from dcl.degree.inventory import ROOT
from dcl.degree.run import suite_for
from dcl.degree.teachers import EXTENSIBLE
from dcl.degree.adapter import FrozenAdapter
from dcl.degree.evaluation import evaluate
from dcl.samplecomplexity.model import QueryPolicy
from dcl.study.storage import atomic_json,file_hash


if __name__=='__main__':
    torch.set_num_threads(1)
    inventory=json.loads((ROOT/'checkpoint-inventory.json').read_text())
    plan=json.loads((ROOT/'confirmatory/design.json').read_text())
    condition=next(c for c in plan['conditions'] if c['id']=='global-n64-d10-mixture1000')
    records=[];tick=time.perf_counter()
    for teacher in EXTENSIBLE:
        for architecture in ('normalized','deepsets','transformer'):
            cp=next(r for r in inventory if r['primary'] and r['teacher']==teacher and r['architecture']==architecture)
            assert file_hash(cp['path'])==cp['sha256']
            suite=suite_for(plan,condition,cp['seed']%100-1)[:32]
            reference,_,_=evaluate(None,teacher,suite,'global',batch=16,diagnostics=False)
            assert all(r['success'] for r in reference)
            model=QueryPolicy(teacher,architecture).eval()
            model.load_state_dict(torch.load(cp['path'],weights_only=False)['model'])
            rows,_,_=evaluate(FrozenAdapter(model),teacher,suite,'global',batch=16)
            expected=json.loads((ROOT/'confirmatory'/cp['id']/condition['id']/'rows.json').read_text())[:32]
            matches=sum(a==b for a,b in zip(rows,expected))
            assert matches==32,(cp['id'],matches)
            records.append(dict(checkpoint=cp['id'],rows=32,identical_rows=matches,successes=sum(r['success'] for r in rows)))
    result=dict(checkpoints=len(records),replayed_rows=sum(r['rows'] for r in records),
        identical_rows=sum(r['identical_rows'] for r in records),records=records,wall_seconds=time.perf_counter()-tick,
        new_independent_final_samples=0,scientific_training_updates=0,
        interpretation='Read-only reproduction of existing final cases, including the Johansson cutoff; never replaces or enlarges a final battery.')
    atomic_json(ROOT/'reproduction-smoke.json',result)
    print(json.dumps(result))
