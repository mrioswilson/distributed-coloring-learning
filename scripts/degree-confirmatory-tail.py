#!/usr/bin/env python3
"""Compute the unstarted tail of the same frozen design on spare CPU cores.

The original executor later verifies/reuses these immutable completed rows.
No instance, checkpoint, batch, precision, or inference source is changed.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import time
from pathlib import Path
from dcl.degree.inventory import ROOT
from dcl.degree.run import worker, source_hashes
from dcl.study.storage import atomic_json, file_hash


if __name__ == "__main__":
    plan=json.loads((ROOT/"confirmatory/design.json").read_text())
    assert source_hashes()==plan['sources']
    assert (ROOT/'confirmatory/exact-validation.json').exists()
    cps=json.loads((ROOT/'checkpoint-inventory.json').read_text())
    cps+=json.loads((ROOT/'supplemental-checkpoint-inventory.json').read_text())
    cps=[r for r in cps if r['id'] in plan['checkpoints']]
    tail=cps[-60:]
    assert all(not (ROOT/'confirmatory'/r['id']).exists() for r in tail), 'Tail must be unstarted'
    atomic_json(ROOT/'confirmatory/execution-scheduling-amendment.json',dict(
        created_utc=datetime.now(timezone.utc).isoformat(), frozen_design_sha256=file_hash(ROOT/'confirmatory/design.json'),
        reason='Use four additional idle CPU cores for the last sixty unstarted jobs; original four-worker executor continues its prefix.',
        checkpoints=[r['id'] for r in tail], max_simultaneous_inference_workers=8, threads_per_worker=1,
        unchanged='All conditions, graph/tape seeds, sample counts, weights, float32 inference, batches, horizons and source hashes remain frozen.',
        reuse='Original executor reads each completed summary and verifies the raw-row hash. No completed final is rerun or replaced.',
        scheduling_only=True))
    jobs=[('confirmatory',cp,plan,[c for c in plan['conditions'] if not c.get('primary_only') or cp['primary']]) for cp in tail]
    tick=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=[]
        for x in pool.map(worker,jobs):
            results.extend(x)
            print(x[0]['checkpoint_id'],sum(r['successes'] for r in x),sum(r['executions'] for r in x),flush=True)
    atomic_json(ROOT/'confirmatory/tail-execution-receipt.json',dict(wall_seconds=time.perf_counter()-tick,
        summaries=len(results),cpu_seconds=sum(r['cost']['cpu_seconds'] for r in results),
        max_workers=4,new_training_updates=0))
