# UCGNN paper study: measured results and limits

Generated UTC: 2026-10-21T02:57:23.372504+00:00

Protocol: commit 023deaf; primary checkpoint epoch 120. Historical results remain separate.

Training runs present: 8; completed: 7.
Evaluation cases recorded: 0.

## Training

treatment,seed,epoch,parameters,updates,validation_candidate,validation_commit
deepsets,41000,120,17907,17880,0.9948050638018863,1.0
mean,41000,120,16035,17880,0.9982851666918848,1.0
normalized,41000,120,16035,17880,0.9970746961214505,1.0
normalized,41001,40,16035,5880,0.9978001099945003,1.0
own_only,41000,120,16035,17880,0.9970746961214505,1.0
raw_endpoints,41000,120,16035,17880,0.9850204266908761,1.0
sum,41000,120,16035,17880,0.9982851666918848,1.0
transformer,41000,120,16995,17880,0.995712916729712,1.0


## Scope of mathematical claims

Proved: a finite-type MAX reduction and a sufficient conditional progress bound, with complete proofs in `docs/UCGNN_MATHEMATICAL_ANALYSIS.md`.
Verified interval certificates, where present, concern fixed checkpoints and stated arithmetic assumptions; they do not certify candidate diversity or termination.
Measured: all rates in the linked CSVs; candidate-grid laws are quadrature approximations, not exact probabilities.
Hypotheses: necessity of each mechanism outside this fixed training procedure, uniform candidate anti-concentration in every reachable state, and general termination of the learned policy.

## Coverage and statistical limits

`cells.csv` lists planned, executed, algorithmic, technical and unexecuted counts by network/family/degree/N. Confidence intervals apply only to fixed-network random-graph cells.
Paired contrasts resample training seeds and common graph IDs, preserving paired architectures. Five seeds provide limited training-population inference. Pilot-only contrasts are marked.
Cliques are deterministic stress graphs with multiple tapes. Best-selection policy results are separate from epoch 120. No hypothesis-test p-value or selective superiority claim is supplied.

## Reproduce and resume

```bash
bash experiments/ucgnn-paper-study status
bash experiments/ucgnn-paper-study resume
PYTHONPATH=src /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m dcl.paperstudy.report
.tools/tectonic -X compile paper/ucgnn/main.tex
```

The tmux process survives a client disconnect. A container restart requires the explicit resume command.
