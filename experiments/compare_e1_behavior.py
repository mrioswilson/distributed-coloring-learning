#!/usr/bin/env python3
"""Authenticated E1/Luby distribution and same-state local comparisons."""
import argparse
import json
from pathlib import Path

from dcl.e1_behavior import REPO, inspect_inputs, run, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=REPO / "configs/e1_behavior/comparison.json")
    parser.add_argument("--weights-root", type=Path, default=REPO,
                        help="directory containing the data/ucgnn/runs paths from E1_INDEX")
    parser.add_argument("--replication-index", type=Path, help="frozen index of newly trained canonical E1 replicas")
    parser.add_argument("--root", type=Path, default=REPO / "data/e1-behavior-20261021")
    parser.add_argument("--output", type=Path, default=REPO / "results/e1-behavior/2026-10-21")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.check_inputs:
        inputs = inspect_inputs(args.weights_root, args.replication_index)
        print(json.dumps(inputs, indent=2))
        return 0 if inputs["ready"] else 2
    if not args.summarize_only:
        run(args.plan, args.root, args.weights_root, args.replication_index)
    summarize(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
