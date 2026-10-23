# Independent instrument validation before neural confirmation

This artifact is generated before the first new teacher-versus-network bank. It documents method choice, acceptance and limitations separately from later scientific results. V1, V2, development and validation seed namespaces remain distinct.

The final method is the honestly split joint-permutation omnibus V2. Its training-only all-pair search was motivated by separate relocated-pair development failures. All V2 parameters were fixed before second-round validation. No new neural comparison was used to select a kernel, region, checkpoint, sample size or test.

## Calibration

There are 104 method/null cells. Each null law has 400 independent trials at total m=250 per generator. The prespecified family audit uses one-sided binomial exceedance p >= 9.6153846e-05. Minimum observed audit p is 0.0066649326. Passing finite controls is not a universal guarantee; conditional exchangeability and joint permutation ranks provide the level argument. Pointwise intervals may exclude .05 by chance among many cells; the multiplicity-aware audit is reported without hiding them.

| Null law | Omnibus rejections/400 | Wilson 95% interval |
|---|---:|---|
| clique_null | 19/400 | [0.031, 0.073] |
| discrete_null | 15/400 | [0.023, 0.061] |
| point_null | 0/400 | [0.000, 0.010] |
| star_null | 32/400 | [0.057, 0.111] |
| uniform_null | 23/400 | [0.039, 0.085] |
| actual Luby | 10/400 | [0.014, 0.045] |
| actual deepsets-s41000 | 21/400 | [0.035, 0.079] |
| actual normalized-s41000 | 21/400 | [0.035, 0.079] |
| actual transformer-s41000 | 19/400 | [0.031, 0.073] |

instrument-null-calibration.csv and both raw validation tables retain every method; point-mass, synthetic, actual Luby and actual frozen-checkpoint controls are distinguished. All neural self controls use fresh executions, not resampling fitted empirical probabilities.

## Fixed power targets

| Class | Total m per generator | Detections/30 | Lower pointwise Wilson 95% bound |
|---|---:|---:|---:|
| localized | 250 | 30/30 | 0.886 |
| localized_4 | 500 | 30/30 | 0.886 |
| correlation | 1000 | 30/30 | 0.886 |
| higher_order | 1000 | 30/30 | 0.886 |
| diffuse | 1000 | 30/30 | 0.886 |
| correlation_remote | 1000 | 30/30 | 0.886 |

Selection observations are counted in total m. Baselines use all m; split methods reserve half for independent evaluation. These targets concern specified alternatives, not every law with the same TV. Thirty trials have substantial uncertainty.

## All candidate methods on the mandatory star control

| Method | m=100 | 250 | 500 | 1000 | 2000 |
|---|---:|---:|---:|---:|---:|
| old_hamming | 2/30 | 1/30 | 1/30 | 2/30 | 5/30 |
| multi_max | 1/30 | 1/30 | 2/30 | 3/30 | 5/30 |
| multi_softmax | 1/30 | 1/30 | 2/30 | 2/30 | 3/30 |
| omnibus_v1 | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| omnibus | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| local | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| interaction | 5/30 | 17/30 | 30/30 | 30/30 | 30/30 |
| structural | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| split_multiscale | 0/30 | 1/30 | 2/30 | 2/30 | 1/30 |
| split_global | 2/30 | 2/30 | 2/30 | 3/30 | 2/30 |
| classifier | 27/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| multinomial | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |

The star laws are artificial proper-coloring generators, not the actual Luby and neural output laws. Their known TV is 99/199. The old Hamming MMD has population squared effect 1.50213168648e-5 and null SD .000241288 at m=250. The simple center event <=101 has probabilities 100/199 and 1. This exhibits kernel attenuation rather than an information-theoretic inability to distinguish those two known laws.

## Global and higher-order sensitivity, with failures retained

| Class | Old MMD at m=1000 | V1 at m=1000 | V2 at m=1000 |
|---|---:|---:|---:|
| marginal | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] |
| localized_4 | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] |
| correlation | 2/30 [0.02,0.21] | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] |
| correlation_remote | 4/30 [0.05,0.30] | 3/30 [0.03,0.26] | 30/30 [0.89,1.00] |
| higher_order | 2/30 [0.02,0.21] | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] |
| higher_order_remote | 1/30 [0.01,0.17] | 1/30 [0.01,0.17] | 1/30 [0.01,0.17] |
| parity_global | 3/30 [0.03,0.26] | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] |
| structural | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] |
| diffuse | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] |
| large_support | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] | 30/30 [0.89,1.00] |
| rare | 0/30 [0.00,0.11] | 3/30 [0.03,0.26] | 2/30 [0.02,0.21] |

Unknown relocated triples are not covered by an all-pair search. The retained characteristic component ensures fixed-space consistency in ideal arithmetic, not adequate finite power for these high-order alternatives. Rare-event sampling and huge unknown supports give explicit lower bounds; no 80% target is imposed on statistically infeasible cases. Structural summaries are not injective and do not replace the complete-vector component.

## Primary-tail check

| Class | m | B | Threshold | Detections/30 | Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| localized | 250 | 19999 | .05/795 | 30/30 | [0.886,1.000] |
| localized | 500 | 19999 | .05/795 | 30/30 | [0.886,1.000] |
| diffuse | 500 | 19999 | .05/795 | 30/30 | [0.886,1.000] |

The two total-m=500 cells passed the separately fixed >=.80 lower-bound gate. These are conservative first-Holm-threshold checks for named laws, not estimates of power under the entire unknown neural family.

## Compute cost and reproducibility

| Total m | Mean omnibus seconds | Mean combined four-global-kernel seconds |
|---:|---:|---:|
| 100 | 0.1396 | 0.0168 |
| 250 | 0.0951 | 0.0712 |
| 500 | 0.4007 | 0.5359 |
| 1000 | 1.5297 | 3.0624 |
| 2000 | 5.9310 | 13.4561 |

Timing columns measure complete omnibus and combined global-scale calculations; they do not estimate an isolated single-kernel runtime. CPU worker contention is part of measured wall time. Algorithm execution costs have a separate GPU-process ledger and prospective per-cell budget matrix. No neural network was trained. The simple source classifiers are statistical tools trained only within their designated blocks.

The complete validation arrays and hashes, trial-level p-values, seeds and both method versions are archived. All sample sizes, bandwidths, feature families and external multiplicity rules for new model comparisons will be frozen in the separate confirmation protocol. MMDAgg and MMD-FUSE motivated comparisons but are not claimed as exact reproduced algorithms or inherited minimax guarantees.
