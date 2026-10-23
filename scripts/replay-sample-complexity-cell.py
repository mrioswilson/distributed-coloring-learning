#!/usr/bin/env python3
"""Reproduce one primary training from stored labels, without final evaluation."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dcl.samplecomplexity.runner import verify
from dcl.samplecomplexity.training import train
from dcl.study.storage import atomic_json

p=argparse.ArgumentParser()
p.add_argument("--teacher",default="LUBY")
p.add_argument("--architecture",default="normalized")
p.add_argument("--seed",type=int,default=83101)
p.add_argument("--budget",type=int,default=1000)
args=p.parse_args()
config=json.loads(Path("configs/sample-complexity.json").read_text());verify(config)
records=json.loads(Path("results/sample-complexity/training.json").read_text())
expected=next(r for r in records if tuple(r["identity"][k] for k in ("teacher","architecture","seed","budget"))==
              (args.teacher,args.architecture,args.seed,args.budget))
dataset=Path("data/sample-complexity-v1/datasets")/args.teacher.lower()/f"s{args.seed}"
with tempfile.TemporaryDirectory(prefix="sample-complexity-training-replay-") as temporary:
    replay=train(config,args.teacher,args.architecture,args.seed,args.budget,dataset,temporary,dev=False)
result=dict(passed=replay["initial_hash"]==expected["initial_hash"] and replay["weights_hash"]==expected["weights_hash"],
       identity=expected["identity"],fresh_initialization=True,stored_labels_only=True,new_teacher_queries=0,
       final_executions=0,expected_initial_hash=expected["initial_hash"],actual_initial_hash=replay["initial_hash"],
       expected_weights_hash=expected["weights_hash"],actual_weights_hash=replay["weights_hash"],
       note="Numerical training replay; not a new independent training replica or another confirmation attempt. Checkpoint bytes include timing metadata and are not required to match.")
atomic_json("results/sample-complexity/training-replay.json",result)
if not result["passed"]:raise SystemExit("Replayed model weights differ; inspect software/hardware and deterministic settings")
print(json.dumps(result,indent=2))
