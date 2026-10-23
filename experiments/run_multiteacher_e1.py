"""Repeat the seven-native-teacher comparison with E1 and declared adapters."""
import argparse
import json
from pathlib import Path

from dcl.multiteacher_e1.experiment import run
from rooted_multiteacher_cache import resident_roots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/multiteacher_e1/comparison.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Read-only original native-policy-v1 archive")
    parser.add_argument("--trainers", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.root, args.source, args.trainers, args.stage, resident_roots)
