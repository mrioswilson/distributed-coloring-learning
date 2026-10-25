#!/usr/bin/env python3
"""Reuse the historical outward-interval certificate on the selected weights.

Applies only to canonical cores with no native context residuals. A conditional
COMMIT certificate is neither a candidate-policy nor a termination certificate.
"""
import json
import time
import torch
from dcl.degree.inventory import ROOT
from dcl.samplecomplexity.model import QueryPolicy
from dcl.paperstudy.certification import commit_certificate
from dcl.study.storage import atomic_json, file_hash


if __name__ == "__main__":
    torch.set_num_threads(1)
    cps=[r for r in json.loads((ROOT/'checkpoint-inventory.json').read_text()) if r['teacher'] in ('LUBY','JOHANSSON','GREEDY')]
    results=[]
    tick,cpu=time.perf_counter(),time.process_time()
    for cp in cps:
        model=QueryPolicy(cp['teacher'],cp['architecture'])
        model.load_state_dict(torch.load(cp['path'],weights_only=False)['model'])
        assert not model.native, 'Native context is not covered by the core certificate'
        r=dict(checkpoint=cp['id'],teacher=cp['teacher'],architecture=cp['architecture'],seed=cp['seed'],
            sha256=cp['sha256'],real=commit_certificate(model.core),float32_model=commit_certificate(model.core,True))
        atomic_json(ROOT/'commit-certificates'/f"{cp['id']}.json",r)
        results.append(r)
    atomic_json(ROOT/'commit-certificates/index.json',results)
    atomic_json(ROOT/'commit-certificates/cost.json',dict(wall_seconds=time.perf_counter()-tick,cpu_seconds=time.process_time()-cpu,
        checkpoints=len(results),type_supports_per_checkpoint=64,arithmetic_modes=2,
        source_sha256=file_hash('src/dcl/paperstudy/certification.py'),new_training_updates=0))
    print(json.dumps(dict(checkpoints=len(results),exact_rule=sum(r['float32_model']['certified_exact_rule'] for r in results),
        no_false_accept=sum(r['float32_model']['certified_no_false_accept'] for r in results))))
