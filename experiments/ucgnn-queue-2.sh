#!/usr/bin/env bash
# Queue 2 (inside the container): E2/E3 replica 0, their evaluations, A1 extra, checks, benchmark.
set -euo pipefail
experiments/ucgnn run --plan configs/ucgnn/e2.json --root data/ucgnn
experiments/ucgnn run --plan configs/ucgnn/e3.json --root data/ucgnn
R1=data/ucgnn/runs/e1-normalized-counterfactual-s00-90d67ee02dc0
R2=data/ucgnn/runs/e2-normalized-counterfactual-s00-7f6fdc95eda4
R3=data/ucgnn/runs/e3-normalized-counterfactual-s00-ba2c7ae460e0
RA=data/ucgnn/runs/a1-raw_prefix-counterfactual-s00-2a0a9bd42deb
for pair in "$R2 e2-s00" "$R3 e3-s00"; do
    set -- $pair
    experiments/ucgnn-evaluate-view "$1/epoch-120.pt" "$2-epoch-120" 3 5 8 10
    experiments/ucgnn-evaluate-extra "$1/epoch-120.pt" "$2-epoch-120"
    experiments/ucgnn-evaluate-view "$1/best.pt" "$2-best" 3 5 8 10
done
experiments/ucgnn-evaluate-extra "$RA/epoch-120.pt" a1-s00-epoch-120
for pair in "$R1 e1-s00" "$R2 e2-s00" "$R3 e3-s00"; do
    set -- $pair
    experiments/ucgnn permcheck --checkpoint "$1/epoch-120.pt" --output "data/ucgnn/permcheck/$2-epoch-120.json"
done
experiments/ucgnn benchmark --checkpoint "$R1/epoch-120.pt" --output data/ucgnn/benchmark/forward-cost.csv
echo QUEUE2-DONE
