#!/usr/bin/env python3
"""Repeat sealed release audits in a new results directory; never resume them.

The destination must be inside this checkout's results/ tree because the
sealed drivers record paths relative to the checkout. This wrapper does not
modify the drivers, their original freezes or reference outcomes.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"results/martin-artifacts-20261031"


def load(name):
    spec=importlib.util.spec_from_file_location("sealed_release_"+name,ROOT/"scripts"/f"integrated-release-{name}-audit.py")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind",choices=("infer","functional","raw","all"))
    p.add_argument("--destination",required=True,type=Path)
    p.add_argument("--first-checkpoint",action="store_true",help="Technical inference check on one fixed checkpoint, not a scientific endpoint")
    a=p.parse_args();os.chdir(ROOT)
    destination=a.destination.resolve()
    if not destination.is_relative_to(ROOT/"results"):
        raise ValueError("Choose a new directory under this checkout's results/ tree")
    if destination.exists():raise FileExistsError("Existing results are preserved; choose a new destination")
    if a.first_checkpoint and a.kind!="infer":raise ValueError("--first-checkpoint applies only to infer")
    destination.mkdir(parents=True)
    if a.kind in ("infer","functional","all"):
        m=load("model");m.OUT=destination;m.torch.set_num_threads(1)
        shutil.copyfile(REFERENCE/"model-replay-freeze.json",destination/"model-replay-freeze.json")
        m.verify()
        if a.kind=="all":
            m.inventory();m.validate_arrays();m.reanalysis()
        if a.kind in ("infer","all"):
            if a.first_checkpoint:
                fixed_jobs=m.jobs()[:1]
                m.jobs=lambda:fixed_jobs
            m.infer()
        if a.kind in ("functional","all"):
            m.functional()
            actual=json.loads((destination/"functional-replay-audit.json").read_text())
            for record in actual["records"]:
                rel=Path(record["path"]).relative_to(destination.relative_to(ROOT))
                with gzip.open(ROOT/record["path"],"rt") as f:new=json.load(f)
                with gzip.open(REFERENCE/rel,"rt") as f:old=json.load(f)
                assert new["rows"]==old["rows"],str(rel)
    if a.kind in ("raw","all"):
        original_freeze=json.loads((REFERENCE/"raw-audit-freeze.json").read_text())
        for path,digest in {**original_freeze["inputs"],**original_freeze["sources"]}.items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        m=load("raw");m.OUT=destination
        for rel in ("member-index.json","package-metadata/confirmatory-rows/results/final-closure/generalization-cells.csv"):
            target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(REFERENCE/rel,target)
        m.main()
        actual=json.loads((destination/"raw-terminal-audit.json").read_text())
        old=json.loads((REFERENCE/"raw-terminal-audit.json").read_text())
        assert {k:v for k,v in actual.items() if k not in ("seconds","frozen_input_sha256")}=={k:v for k,v in old.items() if k not in ("seconds","frozen_input_sha256")}
    receipts={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in destination.glob("*-audit.json")}
    (destination/"fresh-replay-verification.json").write_text(json.dumps(dict(
        kind=a.kind,first_checkpoint_only=a.first_checkpoint,reference_outcomes_preserved=True,
        all_executed_checks_passed=True,receipts=receipts,
        scope="Fresh execution of fixed consumed evaluations; no new independent test cases"),indent=2)+"\n")


if __name__=="__main__":main()
