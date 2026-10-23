#!/usr/bin/env python3
"""Create an isolated, independent replication without overwriting frozen results."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

repo=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("--directory",type=Path,required=True,help="A new checkout directory that does not already exist")
p.add_argument("--namespace",required=True,help="Independent graph/tape namespace for the replication")
p.add_argument("--seed-offset",type=int,default=0,help="Offset all five training seeds; use a new offset for independent initializations")
p.add_argument("--stages",default="pilot,design,validate,train,controls,confirm,optimization,capacity,law,diagnostics,full_fidelity,failure_witnesses,flops,report,verify")
args=p.parse_args();target=args.directory.resolve()
if target.exists(): raise SystemExit("Replication directory must be new; existing files are never removed")
if not re.fullmatch(r"[A-Za-z0-9_.-]+",args.namespace): raise SystemExit("Use letters, digits, dots, underscores or hyphens for namespace")
original=json.loads((repo/"configs/sample-complexity.json").read_text())
if args.namespace==original["namespace"]: raise SystemExit("Use an independent namespace; the published cohort is replayed with fetch/verify")
stages=args.stages.split(",")
allowed={"pilot","design","validate","train","controls","confirm","optimization","capacity","law","diagnostics","full_fidelity","failure_witnesses","flops","report","verify"}
if not set(stages)<=allowed: raise SystemExit("Unknown stage")
subprocess.run(["git","worktree","add","--detach",str(target),"HEAD"],cwd=repo,check=True)
# Only the brand-new checkout's exported cohort is removed. The original worktree,
# its results, checkpoints and Git history remain untouched.
shutil.rmtree(target/"results/sample-complexity",ignore_errors=True)
for name,suffix in (("sample-complexity.json",""),("sample-complexity-law.json","-law"),
                    ("sample-complexity-capacity.json","-small"),("sample-complexity-optimization.json","-optimization")):
    path=target/"configs"/name
    config=json.loads(path.read_text());config["namespace"]=args.namespace+suffix
    if name=="sample-complexity.json":config["seeds"]=[s+args.seed_offset for s in config["seeds"]]
    path.write_text(json.dumps(config,indent=2)+"\n")
environment=dict(os.environ,PYTHONPATH=str(target/"src"))
for stage in stages:
    if stage in ("pilot","design","validate","train","controls","confirm"):
        command=[sys.executable,"-m","dcl.samplecomplexity.runner",stage]
    elif stage in ("optimization","capacity","law","diagnostics","full_fidelity","failure_witnesses","report"):
        command=[sys.executable,"-m",f"dcl.samplecomplexity.{stage}"]
    elif stage=="flops":command=[sys.executable,"scripts/estimate-sample-complexity-flops.py"]
    else:command=[sys.executable,"scripts/verify-sample-complexity.py","--runtime"]
    print(f"Replication {args.namespace}: {stage}",flush=True)
    subprocess.run(command,cwd=target,env=environment,check=True)
print(f"Independent replication saved at {target}")
