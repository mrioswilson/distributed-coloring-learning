#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/AlgoLearning/ucgnn-local-palette-study
export PYTHONPATH=src OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
STUDY_PY=/workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python
# The first coordinator was already launched. Wait for its process, not just
# its current case, before acquiring the common GPU lock and extending the plan.
STUDY_COORDINATOR_PID=${1:?Pass the existing coordinator PID}
while kill -0 "$STUDY_COORDINATOR_PID" 2>/dev/null; do sleep 20; done
"$STUDY_PY" -m dcl.localpalettes resume
"$STUDY_PY" -m dcl.localpalettes audit
"$STUDY_PY" -m dcl.localpalettes.finalize
"$STUDY_PY" -m dcl.localpalettes.controls
"$STUDY_PY" -m pytest -q --junitxml=results/ucgnn-local-palettes/all-tests.xml > data/ucgnn-local-palettes/final-tests.log 2>&1
/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic -X compile --keep-logs paper/ucgnn/main.tex > data/ucgnn-local-palettes/latex.log 2>&1
"$STUDY_PY" -m dcl.localpalettes status > results/ucgnn-local-palettes/final-status.json
date -u > data/ucgnn-local-palettes/FINAL_STAGE_COMPLETE
