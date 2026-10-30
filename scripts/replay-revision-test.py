#!/usr/bin/env python3
"""Replay frozen tests after relocation without rewriting historical receipts.

The original prospective driver/source freeze is immutable. This companion
maps teacher/seed/arm identities into a restored runtime, checks original
hashes, and writes separate replay comparisons. No training or test selection.
"""
import argparse
import json
from pathlib import Path
import torch
from dcl.paperrevision.experiment import diagnostics, sources
from dcl.multicf.protocol import cases
from dcl.samplecomplexity.training import initialize
from dcl.samplecomplexity.evaluation import evaluate
from dcl.study.storage import file_hash, atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--teacher",choices=("BE_BASE_D3","SPECULATIVE_GM"),required=True)
    parser.add_argument("--seed",type=int,required=True);parser.add_argument("--arm",choices=("native","cf"),required=True)
    args=parser.parse_args();out=Path("results/paper-revision-20261030")
    frozen=json.loads((out/"weights-frozen.json").read_text());assert frozen["sources"]==sources()
    rec=next(r for r in frozen["weights"] if (r["teacher"],r["seed"],r["arm"])==(args.teacher,args.seed,args.arm))
    runtime=args.runtime;directory=runtime/"models"/args.teacher.lower()/f"s{args.seed}"/args.arm
    assert file_hash(directory/"final.pt")==rec["sha256"]
    config=json.loads(Path("configs/paper-revision-20261030.json").read_text())
    model=initialize(args.teacher,"normalized",args.seed).eval()
    model.load_state_dict(torch.load(directory/"final.pt",weights_only=False)["model"])
    keys=set(json.loads((runtime/"datasets"/args.teacher.lower()/f"s{args.seed}"/"type-keys.json").read_text()))
    diagnostic=diagnostics(model,args.teacher,runtime/"test"/args.teacher.lower()/f"s{args.seed}",keys)
    rows,seconds=evaluate(model,args.teacher,cases(config,"final",args.seed))
    original=json.loads((out/"prospective"/args.teacher.lower()/f"s{args.seed}"/(args.arm+"-test.json")).read_text())
    fields=("case_id","colors","failure","native_steps","invariants","terminal_correct")
    same=diagnostic==original["diagnostic"] and len(rows)==len(original["rows"]) and all(
        all(a[k]==b[k] for k in fields) for a,b in zip(rows,original["rows"]))
    result=dict(teacher=args.teacher,seed=args.seed,arm=args.arm,checkpoint_sha256=rec["sha256"],
                same_diagnostics_and_terminal_rows=same,seconds=seconds,rows=len(rows))
    atomic_json(out/"portable-replay"/f"{args.teacher}-{args.seed}-{args.arm}.json",result)
    if not same:raise RuntimeError("Replay differs; preserve both outputs for investigation")
    print(json.dumps(result))


if __name__=="__main__":main()
