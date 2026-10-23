#!/usr/bin/env python3
"""Post hoc dominant-operation FLOP estimates, separate from measured CPU time."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dcl.samplecomplexity.data import load, components
from dcl.samplecomplexity.training import initialize, passive_loss
from dcl.study.storage import atomic_json

out=Path("results/sample-complexity");root=Path("data/sample-complexity-v1")
config=json.loads(Path("configs/sample-complexity.json").read_text())
torch.set_num_threads(1)
calibration=[]
for teacher in config["teachers"]:
    arrays,labels,_=load(root/"pilot/datasets"/teacher.lower(),config["pilot_query_budget"])
    for arch in config["architectures"]:
        for phase in sorted(np.unique(labels[:,0])):
            positions=np.resize(np.flatnonzero(labels[:,0]==phase),config["batch_size"])
            model=initialize(teacher,arch,99101)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as profile:
                loss=passive_loss(model,tuple(a[positions] for a in arrays),labels[positions]);loss.backward()
            events=profile.key_averages()
            flops=int(sum(e.flops for e in events))
            calibration.append(dict(teacher=teacher,architecture=arch,phase=int(phase),
                 batch_size=config["batch_size"],profiled_flops=flops,
                 field_max=int(round(arrays[0][positions,0,15].max()*13)),
                 counted_operations=[dict(name=e.key,flops=int(e.flops)) for e in events if e.flops]))
lookup={(r["teacher"],r["architecture"],r["phase"]):r["profiled_flops"] for r in calibration}
records=json.loads((out/"training.json").read_text());estimates=[]
for t in config["teachers"]:
    for s in config["seeds"]:
        with np.load(root/"datasets"/t.lower()/f"s{s}"/"dataset.npz",allow_pickle=False) as stored: labels=stored["labels"]
        costs=np.asarray([len(components(int(p))) for p in labels[:,0]]);cumulative=np.cumsum(costs)
        for r in records:
            i=r["identity"]
            if i["teacher"]!=t or i["seed"]!=s:continue
            take=np.searchsorted(cumulative,i["budget"],side="right");phases=sorted(map(int,np.unique(labels[:take,0])))
            counts={phase:i["updates"]//len(phases)+int(j<i["updates"]%len(phases)) for j,phase in enumerate(phases)}
            estimate=sum(count*lookup[t,i["architecture"],phase] for phase,count in counts.items())
            estimates.append(dict(**i,dominant_profiled_flops_estimate=estimate,
                                  phase_update_counts=counts,measured_cpu_seconds=r["cpu_seconds"]))
atomic_json(out/"flops-estimates.json",dict(calibration=calibration,training=estimates,
     method="Torch CPU profiler with_flops on batch64 passive forward/backward; multiply by exact cyclic phase update counts",
     limitations="Only profiler-supported matrix/conv arithmetic is counted. Elementwise, reductions, indexing, optimizer, encoder and interpreter are excluded. Linial uses pilot max-field batch as a conservative shape estimate. This is not total FLOPs or measured hardware instruction count.",
     hardware_time_is_separate=True))
print(f"Exported {len(calibration)} phase calibrations and {len(estimates)} training estimates")
