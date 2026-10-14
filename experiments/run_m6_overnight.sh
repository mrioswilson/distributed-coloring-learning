#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/distributed-coloring-learning"
source .venv/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

mkdir -p logs data/results

echo "===== M6 OVERNIGHT START ====="
date

echo "===== MATCHED CONTROL TRAINING ====="

python -u experiments/train_m6_ownonly.py \
  --seed 10001 \
  --epochs 120 \
  --train-graphs 800 \
  --val-graphs 200 \
  --tapes-per-graph 3 \
  --batch-size 128 \
  --warmup 10 \
  2>&1 | tee logs/m6_ownonly_training.log

echo "===== THREE-WAY EVALUATION ====="

python -u experiments/overnight_m6_threeway.py \
  --graphs 500 \
  2>&1 | tee logs/m6_overnight_evaluation.log

echo "===== M6 OVERNIGHT COMPLETE ====="
date
