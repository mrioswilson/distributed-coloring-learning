#!/usr/bin/env python3
"""A separate exploratory size extension for four parallel native teachers."""
from datetime import datetime, timezone
import json
from dcl.degree.inventory import ROOT
from dcl.degree.run import execute, freeze, source_hashes
from dcl.study.storage import file_hash


if __name__=='__main__':
    cps=json.loads((ROOT/'checkpoint-inventory.json').read_text())
    cps=[r for r in cps if r['primary'] and r['teacher'] in ('LUBY','JOHANSSON','COLOR_TRIALS','JP')]
    path=ROOT/'larger-size-pilot/design.json'
    if not path.exists():
        freeze(path,dict(namespace='degree-larger-size-20261025-v1',batch=1,exploratory=True,
            created_utc=datetime.now(timezone.utc).isoformat(),sources=source_hashes(),
            conditions=[dict(id=f'global-n{n}-d{d}-regular',n=n,delta=d,palette='global',families=['regular'],runs=1)
                        for n in (2048,4096) for d in (3,10)],
            checkpoints=[r['id'] for r in cps],checkpoint_hashes={r['id']:r['sha256'] for r in cps},
            interpretation='One run per checkpoint per point, four parallel teachers only. Not an extension of the confirmatory sample or a success certification.',
            serial_limit='GM and Greedy retain the completed n<=1024 exploratory battery; larger serial schedules are not required for the degree contrast. BE remains inapplicable.'))
    plan=json.loads(path.read_text())
    assert source_hashes()==plan['sources']
    execute('larger-size-pilot',plan,cps,4)
