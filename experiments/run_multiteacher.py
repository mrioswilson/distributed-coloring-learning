import argparse
import json
from pathlib import Path

from dcl.multiteacher.experiment import run


def main():
    parser = argparse.ArgumentParser(description="Compare GNN learning of seven native coloring policies")
    parser.add_argument("--plan", type=Path, default=Path("configs/multiteacher/comparison.json"))
    parser.add_argument("--root", type=Path, default=Path("data/multiteacher-20261019"))
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.root)


if __name__ == "__main__":
    main()
