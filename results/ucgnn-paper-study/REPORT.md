# UCGNN paper study: measured results and limits

Generated UTC: 2026-10-21T03:26:55.739474+00:00

Protocol: commit 023deaf; primary checkpoint epoch 120. Historical results remain separate.

Training runs present: 13; completed: 12.
Evaluation cases recorded: 0.

Queue state at report generation: running; accumulated active GPU-process hours: 1.009 / 36.

This report describes recorded coverage, including partial or unexecuted work. The handoff snapshot and exact recovery commands are in `CLOSEOUT.md`.

## Training

treatment,seed,epoch,parameters,updates,validation_candidate,validation_commit
deepsets,41000,120,17907,17880,0.9948050638018863,1.0
mean,41000,120,16035,17880,0.9982851666918848,1.0
normalized,41000,120,16035,17880,0.9970746961214505,1.0
normalized,41001,120,16035,17640,0.9984500774961252,1.0
normalized,41002,120,16035,17760,0.9986499324966248,1.0
normalized,41003,120,16035,17760,0.9984589381586797,1.0
normalized,41004,120,16035,17760,0.9993156376790341,1.0
own_only,41000,120,16035,17880,0.9970746961214505,1.0
raw_endpoints,41000,120,16035,17880,0.9850204266908761,1.0
raw_endpoints,41001,120,16035,17640,0.9897005149742513,1.0
raw_endpoints,41002,88,16035,13024,0.9852492624631232,1.0
sum,41000,120,16035,17880,0.9982851666918848,1.0
transformer,41000,120,16995,17880,0.995712916729712,1.0


## Frozen E1 candidate prediction

CPU float32 grid measurements, with the primary empirical prediction and the two prospective secondary analytic predictions retained separately. Equivalent configured availability patterns are not independent observations.

```csv
palette,predictor,configured_states,mean_state_rank_error,worst_state_mean_rank_error,states_over_two,max_forbidden_grid_rate
17,primary_empirical,30,0.05815845127704569,0.1439726695949243,0,0.0
17,secondary_global,30,0.03424434683585485,0.0707662274280136,0,0.0
17,secondary_context,30,0.04693346347811938,0.084431429965837,0,0.0
63,primary_empirical,30,0.1063120221246136,0.3147877013177159,0,0.0
63,secondary_global,30,0.06458435008947451,0.1961932650073206,0,0.0
63,secondary_context,30,0.051325850008134014,0.0839433870180575,0,0.0
251,primary_empirical,30,0.4214901578005531,1.1957052220595412,0,0.0
251,secondary_global,30,0.2529526598340654,0.915568570034163,0,0.0
251,secondary_context,30,0.08286969253294288,0.1913128355295266,0,0.0
997,primary_empirical,30,1.6900276557670408,4.761346998535871,10,0.0
997,secondary_global,30,1.0236863510655603,3.971205466081015,5,0.0
997,secondary_context,30,0.11965186269725067,0.2362127867252318,0,0.0
```

Per-state availability, entropy, effective support, collision and forbidden-proposal measurements are in `candidate-diagnostics.csv`. Calibrations and preregistration hashes are in `frozen-prediction/`.

## Conditional local COMMIT certificates

The interval results are checkpoint-specific and depend on the explicit architectural and arithmetic assumptions. A certificate that is inapplicable is not an observed safety failure. Full results: `certificate-summary.csv` and `certificates/`.

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
