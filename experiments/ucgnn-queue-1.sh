#!/usr/bin/env bash
# Queue 1 (inside the container): E1 secondary views, ablation A1, E2/E3 pilots.
set -euo pipefail
R1=data/ucgnn/runs/e1-normalized-counterfactual-s00-90d67ee02dc0
experiments/ucgnn-evaluate-view "$R1/best.pt" e1-s00-best 3 5 8 10
experiments/ucgnn-evaluate-extra "$R1/epoch-120.pt" e1-s00-epoch-120
experiments/ucgnn run --plan configs/ucgnn/a1-raw-prefix.json --root data/ucgnn
RA=data/ucgnn/runs/a1-raw_prefix-counterfactual-s00-2a0a9bd42deb
experiments/ucgnn-evaluate-view "$RA/epoch-120.pt" a1-s00-epoch-120 3 5 8 10
experiments/ucgnn run --plan configs/ucgnn/p2-p3-pilots.json --root data/ucgnn
# Pilots: technical check on their training degrees only (separate pilot cohort).
experiments/ucgnn evaluate --checkpoint data/ucgnn/runs/p2-normalized-counterfactual-s00-40b592f16cd6/epoch-020.pt \
    --degrees 3 10 --sizes 40 160 640 2560 --count 10 --split ucgnn_pilot --namespace 777 --teacher-rounds
experiments/ucgnn evaluate --checkpoint data/ucgnn/runs/p3-normalized-counterfactual-s00-ef3180b70b35/epoch-020.pt \
    --degrees 3 5 8 10 --sizes 40 160 640 2560 --count 10 --split ucgnn_pilot --namespace 777 --teacher-rounds
echo QUEUE1-DONE
