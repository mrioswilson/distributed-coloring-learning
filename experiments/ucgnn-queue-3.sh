#!/usr/bin/env bash
# Queue 3 (inside the container): replicas 1-4, Own-only control, full-budget
# secondaries, and E1 intermediate epochs (post hoc diagnostic, no selection).
set -euo pipefail
ids() { .venv/bin/python -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['ids']))" "$1"; }
view() { local id=$1; IFS=- read -r name _ method seed _ <<< "$id"; echo "$name-$seed"; }
evaluate_run() {
    local id=$1 v; v=$(view "$id")
    experiments/ucgnn-evaluate-view "data/ucgnn/runs/$id/epoch-120.pt" "$v-epoch-120" 3 5 8 10
    experiments/ucgnn-evaluate-extra "data/ucgnn/runs/$id/epoch-120.pt" "$v-epoch-120"
}
experiments/ucgnn run --plan configs/ucgnn/c1-own-only.json --root data/ucgnn
for id in $(ids configs/ucgnn/c1-own-only.json); do evaluate_run "$id"; done
experiments/ucgnn run --plan configs/ucgnn/replication.json --root data/ucgnn
for id in $(ids configs/ucgnn/replication.json); do evaluate_run "$id"; done
experiments/ucgnn run --plan configs/ucgnn/secondary-full-budget.json --root data/ucgnn
for id in $(ids configs/ucgnn/secondary-full-budget.json); do evaluate_run "$id"; done
R1=data/ucgnn/runs/e1-normalized-counterfactual-s00-90d67ee02dc0
for epoch in 010 020 040 060 080 100; do
    experiments/ucgnn-evaluate-view "$R1/epoch-$epoch.pt" "e1-s00-epoch-$epoch" 3 5 8 10
done
echo QUEUE3-DONE
