#!/usr/bin/env python3
"""Numerically replay BE's fixed-label optimization result; extend to 3 cores."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path

import torch

from dcl.samplecomplexity.training import train
from dcl.study.storage import atomic_json, file_hash

ORIGINAL=Path("/workspaces/AlgoLearning/ucgnn-sample-complexity/data/sample-complexity-v1")
OUT=Path("results/multiteacher-counterfactual/be-fixed-label-replay")


def job(args):
    architecture,seed=args
    config=json.loads(Path("configs/sample-complexity.json").read_text())
    dataset=ORIGINAL/"datasets/be_base_d3"/f"s{seed}"
    record=train(config,"BE_BASE_D3",architecture,seed,3000,dataset,OUT/architecture/f"s{seed}",updates=1500)
    if architecture=="normalized":
        original=json.loads((ORIGINAL/"controls/optimization/be_base_d3"/f"s{seed}"/"record.json").read_text())
        replay=dict(seed=seed,architecture=architecture,
            original_checkpoint_sha256=original["checkpoint_sha256"],new_checkpoint_sha256=record["checkpoint_sha256"],
            initial_match=record["initial_hash"]==original["initial_hash"],
            data_prefix_match=record["accounting"]["prefix_sha256"]==original["accounting"]["prefix_sha256"],
            tensor_weights_match=record["weights_hash"]==original["weights_hash"],
            status="Numerical replay, not an independent training-seed replication")
        atomic_json(OUT/architecture/f"s{seed}"/"historical-replay.json",replay)
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--original-runtime",type=Path,default=ORIGINAL)
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args();ORIGINAL=args.original_runtime;OUT=args.output
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(job,[(a,s) for a in ("normalized","deepsets","transformer") for s in range(83101,83106)]))
    atomic_json(OUT/"records.json",rows)
