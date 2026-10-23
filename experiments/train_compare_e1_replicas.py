#!/usr/bin/env python3
"""Train, freeze, compare and audit five newly trained canonical E1 replicas."""
import argparse
import json

from dcl.e1_replication import ROOT, OUTPUT, prepare_training, train_parallel, freeze, compare, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "train", "freeze", "compare", "audit", "all"), default="all")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare_training()
    if args.stage in ("train", "all"):
        train_parallel(args.seeds, workers=args.workers)
    if args.stage in ("freeze", "all"):
        freeze()
    if args.stage in ("compare", "all"):
        compare()
    if args.stage in ("audit", "all"):
        print(json.dumps(audit(), indent=2), flush=True)


if __name__ == "__main__":
    main()
