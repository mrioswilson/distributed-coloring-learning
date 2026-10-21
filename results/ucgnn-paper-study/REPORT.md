# UCGNN paper study: measured results and limits

Generated UTC: 2026-10-21T11:05:26.965303+00:00

Protocol: commit 023deaf; primary checkpoint epoch 120. Historical results remain separate.

Training runs present: 32; completed: 32.
Evaluation cases recorded: 18363.

Queue state at report generation: complete; accumulated active GPU-process hours: 6.172 / 36.

This report describes recorded coverage, including partial or unexecuted work. The handoff snapshot and exact recovery commands are in `CLOSEOUT.md`.

## Training

treatment,seed,epoch,parameters,updates,validation_candidate,validation_commit
cyclic_teacher,41005,120,16035,17760,0.9902893478582343,1.0
deepsets,41000,120,17907,17880,0.9948050638018863,1.0
deepsets,41001,120,17907,17640,0.9964501774911254,1.0
deepsets,41002,120,17907,17760,0.9969998499924996,1.0
deepsets,41003,120,17907,17760,0.9881189103201432,1.0
deepsets,41004,120,17907,17760,0.9934985579508236,1.0
mean,41000,120,16035,17880,0.9982851666918848,1.0
mean,41001,120,16035,17640,0.9980500974951253,1.0
mean,41002,120,16035,17760,0.99949997499875,1.0
mean,41003,120,16035,17760,0.9980115331079737,1.0
mean,41004,120,16035,17760,0.9994134037248863,1.0
normalized,41000,120,16035,17880,0.9970746961214505,1.0
normalized,41001,120,16035,17640,0.9984500774961252,1.0
normalized,41002,120,16035,17760,0.9986499324966248,1.0
normalized,41003,120,16035,17760,0.9984589381586797,1.0
normalized,41004,120,16035,17760,0.9993156376790341,1.0
own_only,41000,120,16035,17880,0.9970746961214505,1.0
raw_endpoints,41000,120,16035,17880,0.9850204266908761,1.0
raw_endpoints,41001,120,16035,17640,0.9897005149742513,1.0
raw_endpoints,41002,120,16035,17760,0.9891494574728736,1.0
raw_endpoints,41003,120,16035,17760,0.9882183336647444,1.0
raw_endpoints,41004,120,16035,17760,0.9886102556582099,1.0
sum,41000,120,16035,17880,0.9982851666918848,1.0
sum,41001,120,16035,17640,0.9980000999950003,1.0
sum,41002,120,16035,17760,0.9990999549977498,1.0
sum,41003,120,16035,17760,0.9982103797971764,1.0
sum,41004,120,16035,17760,0.9992667546561079,1.0
transformer,41000,120,16995,17880,0.995712916729712,1.0
transformer,41001,120,16995,17640,0.992450377481126,1.0
transformer,41002,120,16995,17760,0.9961998099904995,1.0
transformer,41003,120,16995,17760,0.9931895008948101,1.0
transformer,41004,120,16995,17760,0.9969203695556533,1.0


## Primary epoch-120 outcomes

```csv
treatment,networks,executions,successes,seed_rate_min,seed_rate_max,unsafe,monochromatic_edges,unfinished,pending,rounds_max
normalized,5,1520,1520,1.0,1.0,0,0,0,0,18
raw_endpoints,5,1520,1174,0.7664473684210527,0.7796052631578947,0,0,346,78356,256
sum,5,1520,324,0.14473684210526316,0.23026315789473684,0,0,1196,224659,256
mean,5,1520,681,0.41118421052631576,0.4868421052631579,57340,38173,0,0,18
deepsets,5,1520,1520,1.0,1.0,0,0,0,0,26
transformer,5,1520,1520,1.0,1.0,0,0,0,0,49
```

Detailed seed, family, degree and size coverage is in `cells.csv`; `overall-results.csv` contains descriptive totals only. Paired bootstrap intervals are empirical resampling intervals: all-equal outcomes can yield degenerate endpoints and do not establish population certainty.

## Evaluation inventory (descriptive totals, no pooled binomial inference)

kind,treatment,selection,executions,networks,successes,unsafe,pending
historical,A1,epoch120,304,1,265,0,14120
historical,C1,epoch120,304,1,304,0,0
historical,E1,epoch120,1520,5,1520,0,0
paper,cyclic_teacher,epoch120,304,1,304,0,0
paper,deepsets,best,960,5,960,0,0
paper,deepsets,epoch120,1520,5,1520,0,0
paper,mean,best,960,5,494,21090,0
paper,mean,epoch120,1520,5,681,57340,0
paper,normalized,best,960,5,960,0,0
paper,normalized,epoch120,1520,5,1520,0,0
paper,own_only,best,192,1,192,0,0
paper,own_only,epoch120,304,1,304,0,0
paper,raw_endpoints,best,960,5,799,0,33012
paper,raw_endpoints,epoch120,1520,5,1174,0,78356
paper,sum,best,960,5,255,0,87265
paper,sum,epoch120,1520,5,324,0,224659
paper,transformer,best,960,5,960,0,0
paper,transformer,epoch120,1520,5,1520,0,0
specialist,M6B,epoch120,555,15,555,0,0


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
