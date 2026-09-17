#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/m4b-seeds

SEEDS=(
    1234
    2001
    3001
    4001
    5001
    6001
    7001
    8001
    9001
    10001
)

for SEED in "${SEEDS[@]}"; do

    CHECKPOINT="data/results/m4b_seed${SEED}_n40.pt"
    RESULT="data/results/m4b_seed${SEED}_extrapolation.csv"

    echo
    echo "======================================"
    echo "M4B SEED $SEED"
    echo "======================================"

    if [[ -f "$CHECKPOINT" ]]; then
        echo "checkpoint already exists: $CHECKPOINT"
    else
        python experiments/train_m4_seed.py \
            --arch B \
            --seed "$SEED" \
            2>&1 | tee \
            "logs/m4b-seeds/train_${SEED}.log"
    fi

    if [[ -f "$RESULT" ]]; then
        echo "evaluation already exists: $RESULT"
    else
        python experiments/evaluate_m4_seed.py \
            --arch B \
            --seed "$SEED" \
            2>&1 | tee \
            "logs/m4b-seeds/eval_${SEED}.log"
    fi

done

echo
echo "ALL REQUESTED SEEDS COMPLETE"
