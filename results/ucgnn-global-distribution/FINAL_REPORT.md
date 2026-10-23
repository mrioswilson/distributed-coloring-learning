# Global complete-coloring distributions: final technical and statistical report

No new networks were trained. Historical files were read only. No remote push was performed.

The prospective primary family contains **795 tests on 53 fixed graph entries and 15 frozen policies**, with 250 fresh complete runs per generator per graph. **183/795** reject after Holm correction at familywise alpha 0.05. These are conditional-on-valid-completion comparisons. Nonrejection does not prove equality.

| Architecture | Tests | Holm rejections | Nonrejections | Empirical MMD range across graphs |
|---|---:|---:|---:|---:|
| NORMALIZED (E1) | 265 | 10 | 255 | 0.0321–0.1664 |
| DeepSets | 265 | 46 | 219 | 0.0291–0.1799 |
| LocalPaletteTransformer | 265 | 127 | 138 | 0.0287–0.2873 |

There are 53 planned graph entries but 51 unique labeled graphs: star3/star-d3 and wheel-d3/clique-d3 are duplicate graph hashes with fresh banks. They are not independent graph replicates. The bipartite-n256-d3 entry has actual maximum degree 4, correctly recorded in the frozen graph manifest; its suffix is the generation target, not its measured degree. The MMD ranges above describe this finite battery. They are not universally comparable normalized distances across different graphs. Every individual test, effect interval and adjusted p-value is in `primary.json` and `primary.csv`.

## Execution and success

The audit verified 1013 banks and **937,600 distinct execution seeds**. Status totals [valid, invalid-complete, cutoff, technical] = [937600, 0, 0, 0]. Charged GPU-process time, including pilot, teacher sampling and lock-held overhead: **2.350 / 12 hours**.

Every final vector was checked against all edges and its own local list. The frozen model hashes match the recent paper-study cohort (seeds 41000–41004, epoch 120), not the historical E1 cohort. TF32 was disabled, float32 neural input conversion is explicit, and the teacher uses float64 PCG64 uniforms. Actual seed-41000 policies and all fifteen CPU integration tests reproduced hard actions across reset/interleaving and batching probes. This is numerical validation, not a proof that every boundary input is invariant.

## Calibration controls

| Control | Rejections / trials | 95% rate interval |
|---|---:|---|
| Luby/Luby | 6/100 | [0.028, 0.125] |
| known distinct valid point masses | 100/100 | [0.963, 1.000] |
| same checkpoint | 22/450 | [0.033, 0.073] |

Null control groups are compatible with their preregistered 99% binomial acceptance intervals. The two point-mass control vectors (0,1,0) and (1,2,1) have identical color counts, class sizes and unlabeled partitions, yet different labeled laws. A point-mass positive control is intentionally easy; it cannot establish sensitivity against every neural alternative. Teacher/teacher controls include a high-dimensional degree-100 graph. Same-checkpoint controls cover all fifteen policies on path3 only.

## Exact small graphs

All seven supports were enumerated and teacher laws computed using rational absorbing-chain recursion, independently of Monte Carlo estimation banks. The path3 probabilities are 343/1628 (two vectors), 205/1628 (two), and 133/814 (two), rather than 1/6. Uniform proposals therefore do not imply uniformly distributed final solutions. This also occurs with homogeneous lists: on cycle4, each two-color vector has mass 35051/859584 and each three-color vector 36071/573056, giving TV 38111/429792 from the uniform valid-coloring law. `exact-*.json` archives every rational mass and reachable-state count.

The exact reference is the teacher eventual-absorption law; the empirical banks are conditional on valid completion by 128 rounds. A conservative per-node progress argument bounds their teacher-reference TV difference by n(1-2^(-Delta-1))^128; each small-graph row reports this bound. This distinction is retained even with zero observed cutoffs. The 2000-run independent neural estimation banks give plug-in TV against exact teacher probabilities and finite-support concentration intervals in `small-tv.csv`. TV estimation has sampling bias; intervals, rather than raw point estimates alone, govern interpretation. Exact teacher versus uniform is a separate reference. Teacher Monte Carlo validation is reported in the same table; finite nonzero empirical TV is expected.

## Statistically supported differences

| Graph | Policy | U-statistic MMD² | Raw p | Holm p |
|---|---|---:|---:|---:|
| bipartite-n256-d100 | deepsets-s41000 | 0.000915793 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | deepsets-s41001 | 0.000680577 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | deepsets-s41002 | 0.00446594 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | deepsets-s41003 | 0.00557919 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | deepsets-s41004 | 0.0042584 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | normalized-s41000 | 0.00041801 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | normalized-s41001 | 0.000220629 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | normalized-s41002 | 0.000307348 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | transformer-s41000 | 0.00411219 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | transformer-s41001 | 0.0102035 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | transformer-s41002 | 0.0168778 | 5e-05 | 0.03975 |
| bipartite-n256-d100 | transformer-s41003 | 0.0151792 | 5e-05 | 0.03975 |

This table shows at most twelve examples; `primary.csv` contains every result, including all negative results. The complete list of rejected and nonrejected cells is also exported separately. A valid coloring in both groups is not evidence of equal output laws.

## Answers to the ten scientific questions

**P1. Luby versus NORMALIZED (E1):** yes, in 10 of the 265 fixed graph/checkpoint contrasts after Holm correction; this conclusion concerns precisely the cells in `rejected.csv`.

**P2. Luby versus DeepSets:** yes, in 46 of the 265 fixed graph/checkpoint contrasts after Holm correction; this conclusion concerns precisely the cells in `rejected.csv`.

**P3. Luby versus LocalPaletteTransformer:** yes, in 127 of the 265 fixed graph/checkpoint contrasts after Holm correction; this conclusion concerns precisely the cells in `rejected.csv`.

**P4. Graphs, degrees and sizes:** `by-degree-family-size.csv` reports all covered strata; `primary.csv` retains exact graph/checkpoint identities. Regular graphs cover all eight requested degrees at both n=256 and n=512; heterogeneous families extend through degree 200. Stars and cliques have size dictated by degree. One graph per random grid cell limits graph-seed generalization. The derived graph-structure.csv reports connectivity and density; degree-2 random regular graphs can be disconnected. The initial resource plan used coarse pilot extrapolation rather than a separately archived per-cell forecast; resource-costs.csv retains all realized cell costs, and the hard budget was enforced independently of test outcomes.

**P5. Does distinguishability increase with degree at fixed size and family?** Figure 2 and `controlled-degree-changes.csv` describe the fixed-n regular grids separately for every policy. We do not claim a universal monotone relationship: kernel geometry changes with the graph, only one graph is sampled per cell, and degree is not randomized independently of graph construction. Figure 2 and the adjacent changes use unbiased MMD-squared estimates to avoid a degree-dependent positive finite-sample bias of the nonnegative empirical norm; negative estimates are permitted. Descriptive signs and variability are retained rather than replacing them with an unplanned pooled regression.

**P6. How many independent complete colorings?** Primary detections use 250 per generator. Figure 3 and `power.csv` examine 100, 250, 500, 1000 and 2000 for three prespecified graphs and seed 41000 of each architecture. These are conditional empirical-bank power estimates at nominal alpha .05; their 30-trial intervals omit finite-bank uncertainty and do not estimate Holm-adjusted population power. They cannot justify a universal required sample size. Figures 3b/3c and independent-power.csv separately estimate repeated-experiment population power for the full path3 grid and for m=100,250 on star-d200 and regular-n256-d100, using twenty independent trials per point. These nominal-alpha rates do not establish Holm-adjusted power or power for the other four training checkpoints.

**P7. Variation among independent trainings:** Figure 6 and `by-training-seed.csv` show each of the five fixed policies separately. These are five independently trained functions, never a single mixed output distribution. Variation between policies remains distinct from Monte Carlo intervals within each policy. Shared graphs are blocks; no independent graph-by-seed population standard error is claimed.

**P8. Different proposals but indistinguishable complete outputs?** Yes for the ideal equivariant protocols on cliques: conditional on proper completion, vertex-permutation symmetry forces the uniform law on all n! valid colorings for every fixed checkpoint and for Luby. This is a mathematical sufficient-condition result, not an inference from nonrejection. Initially all-free K=201 proposal discrepancies also apply to any node of clique K201 because its deterministic MAX first-stage input matches the archived star-center witness. Floating-point implementation equivariance has numerical probes rather than a universal bitwise certificate. Earlier proposal tests alone cannot answer other graphs. This extension archives a contextual, explicitly post-hoc match between the previous all-free center law at K=201 and the identical initial degree-200 star, if available. Any nonrejection there means failure to detect a global difference at m=250, not equality or cancellation of all proposal differences. See `proposal-context.csv`; conditional proposals along entire trajectories were not exhaustively measured.

**P9. Structural explanations:** Figure 5 and `structural.json` report color count, class concentration/entropy, degree-wise color masses, nonadjacent co-color probabilities and partition diversity. Differences provide interpretable correlates, not causal explanations or a replacement for the labeled-vector test. Equal color-count laws would not imply equal complete laws. Partition encodings deliberately discard color names and are secondary.

**P10. Paper conclusions:** The provisional manuscript section reports exact teacher nonuniformity, precisely enumerated Holm rejections and negative-result limits. Observed differences show that imitation can produce different solution laws on these finite graphs. No observation proves universal inequality, equality, termination, monotonic degree dependence or equivalence over all graphs.

## Uncertainty and limitations

Population MMD intervals use a conservative distribution-free RKHS mean bound. They are pointwise, may be wide, and are not exact TV distances on large spaces. Approximate equivalence tolerance MMD<=0.02 was specified prospectively; cells satisfying the upper-bound criterion: 0. Failure of this criterion is not evidence of inequivalence. With only 19,999 permutations, p-values have finite resolution; strong alternatives can share the minimum raw p. Holm depends on the whole planned family, not only detected cases.

Success-frequency intervals and unconditional/coarsened failure tests are separate artifacts. If all runs succeed, this records zero observed failures rather than probability-zero failure. No training, calibration or classifier test split is reused. Power resampling repeats may coincidentally draw an identical vector, which is allowed by the empirical law and does not merge RNG streams.

## Files and reproduction

See `GLOBAL_DISTRIBUTIONS.md` at the worktree root for exact commands. `sampling-manifest.json` contains all raw-bank paths and SHA256 hashes; `audit.json` checks every archive and seed, and replays three primary tests. `freeze.json` records the immutable protocol and checkpoint/graph hashes. `docs/GLOBAL_COLORING_DISTRIBUTIONS_THEORY.md` gives full finite-space proofs and the verified primary MMD reference. The six main and two supplementary PDF/PNG figures regenerate from archived data.

Clique symmetry diagnostic: 0/105 primary clique-entry/checkpoint tests reject after Holm. The mathematical symmetry result concerns ideal equivariant conditional laws; finite numerical probes and tests do not constitute a universal hardware-level symmetry certificate. All these cells are retained in clique-symmetry-check.csv.


## Observed degree contrasts and structural diagnostics

DeepSets, regular n=256: 16/35 adjacent-degree unbiased MMD-squared estimate changes are positive across the five separately plotted policies. These signs are descriptive estimates, without a monotonic population claim.
DeepSets, regular n=512: 17/35 adjacent-degree unbiased MMD-squared estimate changes are positive across the five separately plotted policies. These signs are descriptive estimates, without a monotonic population claim.
NORMALIZED (E1), regular n=256: 19/35 adjacent-degree unbiased MMD-squared estimate changes are positive across the five separately plotted policies. These signs are descriptive estimates, without a monotonic population claim.
NORMALIZED (E1), regular n=512: 19/35 adjacent-degree unbiased MMD-squared estimate changes are positive across the five separately plotted policies. These signs are descriptive estimates, without a monotonic population claim.
LocalPaletteTransformer, regular n=256: 15/35 adjacent-degree unbiased MMD-squared estimate changes are positive across the five separately plotted policies. These signs are descriptive estimates, without a monotonic population claim.
LocalPaletteTransformer, regular n=512: 15/35 adjacent-degree unbiased MMD-squared estimate changes are positive across the five separately plotted policies. These signs are descriptive estimates, without a monotonic population claim.

Degree-wise color frequencies and nonadjacent co-color comparisons are in structural-comparisons.csv. Their point estimates aggregate full-run features; vertices are never used as independent units for uncertainty. The following rejected cells have the largest observed degree-weighted color-frequency TV projections (descriptive, not complete-vector TV or causal attribution).

| Graph | Policy | Empirical degree-color projection TV | Mean used-color shift |
|---|---|---:|---:|
| bipartite-n256-d200 | transformer-s41001 | 0.3217 | -10.444 |
| random-n256-d200 | transformer-s41004 | 0.2655 | -4.072 |
| bipartite-n256-d100 | transformer-s41001 | 0.2570 | 1.588 |
| star3 | transformer-s41000 | 0.2530 | -0.188 |
| random-n256-d200 | transformer-s41002 | 0.2500 | 1.120 |
| random-n256-d200 | transformer-s41003 | 0.2488 | 1.784 |
| star-d20 | transformer-s41000 | 0.2453 | -0.024 |
| bipartite-n256-d100 | transformer-s41002 | 0.2451 | 2.616 |

| Graph | Policy | Mean used-colors difference (model minus Luby) |
|---|---|---:|
| regular-n512-d200 | transformer-s41004 | -14.756 |
| regular-n256-d200 | transformer-s41004 | -13.432 |
| bipartite-n256-d200 | transformer-s41001 | -10.444 |
| regular-n256-d200 | transformer-s41001 | -8.536 |
| bipartite-n256-d200 | transformer-s41002 | -6.688 |
| regular-n256-d100 | transformer-s41004 | -6.364 |
| bipartite-n256-d200 | transformer-s41004 | -5.180 |
| bipartite-n256-d200 | deepsets-s41004 | -4.724 |

These are observed run-level mean shifts, not separately calibrated structural discoveries or causal mechanisms.

## Independent population-power supplement

The prospective supplement in `docs/GLOBAL_COLORING_POWER_SUPPLEMENT.md` fixes a new 77,000-run bank per generator on path3, split into disjoint blocks for twenty independent experiments at each m. Its rates estimate population detection probability for the specified policies and graphs. Teacher blocks are shared across policies but never across trials or m. Figure 3b and `independent-power.csv` report broad binomial intervals; Independent population power at m=100 and 250 is also measured on star-d200 and regular-n256-d100; higher m on those two graphs remain conditionally estimated.

| Graph | Policy | m | Rejections / 20 | 95% population-power interval |
|---|---|---:|---:|---|
| path3 | normalized-s41000 | 100 | 1/20 | [0.009, 0.236] |
| path3 | normalized-s41000 | 250 | 1/20 | [0.009, 0.236] |
| path3 | normalized-s41000 | 500 | 1/20 | [0.009, 0.236] |
| path3 | normalized-s41000 | 1000 | 1/20 | [0.009, 0.236] |
| path3 | normalized-s41000 | 2000 | 3/20 | [0.052, 0.360] |
| path3 | deepsets-s41000 | 100 | 0/20 | [0.000, 0.161] |
| path3 | deepsets-s41000 | 250 | 0/20 | [0.000, 0.161] |
| path3 | deepsets-s41000 | 500 | 4/20 | [0.081, 0.416] |
| path3 | deepsets-s41000 | 1000 | 1/20 | [0.009, 0.236] |
| path3 | deepsets-s41000 | 2000 | 2/20 | [0.028, 0.301] |
| path3 | transformer-s41000 | 100 | 16/20 | [0.584, 0.919] |
| path3 | transformer-s41000 | 250 | 20/20 | [0.839, 1.000] |
| path3 | transformer-s41000 | 500 | 20/20 | [0.839, 1.000] |
| path3 | transformer-s41000 | 1000 | 20/20 | [0.839, 1.000] |
| path3 | transformer-s41000 | 2000 | 20/20 | [0.839, 1.000] |
| star-d200 | normalized-s41000 | 100 | 2/20 | [0.028, 0.301] |
| star-d200 | normalized-s41000 | 250 | 4/20 | [0.081, 0.416] |
| star-d200 | deepsets-s41000 | 100 | 0/20 | [0.000, 0.161] |
| star-d200 | deepsets-s41000 | 250 | 2/20 | [0.028, 0.301] |
| star-d200 | transformer-s41000 | 100 | 20/20 | [0.839, 1.000] |
| star-d200 | transformer-s41000 | 250 | 20/20 | [0.839, 1.000] |
| regular-n256-d100 | normalized-s41000 | 100 | 6/20 | [0.145, 0.519] |
| regular-n256-d100 | normalized-s41000 | 250 | 11/20 | [0.342, 0.742] |
| regular-n256-d100 | deepsets-s41000 | 100 | 3/20 | [0.052, 0.360] |
| regular-n256-d100 | deepsets-s41000 | 250 | 12/20 | [0.387, 0.781] |
| regular-n256-d100 | transformer-s41000 | 100 | 4/20 | [0.081, 0.416] |
| regular-n256-d100 | transformer-s41000 | 250 | 18/20 | [0.699, 0.972] |

P8 contextual result: 7 of fifteen star/checkpoint comparisons have a previous initial-center proposal-TV lower bound above zero and no Holm rejection of the complete-vector law here. This is a post-hoc match of the identical initial state, not evidence of equal global laws. The complete fifteen-row comparison and original source hash are in proposal-context.csv.

## Known localized alternative: a finite-power failure

A separately prospective supplementary control holds all 200 star leaves iid uniform on {0,1}; only the center law changes, from uniform on 199 high colors to uniform on 100. Every vector is a valid local-list coloring. Exact global TV is 99/199 (about 0.4975), whereas exact MMD squared is 1.5021316864801976e-05. The primary kernel is unchanged. This control demonstrates why characteristicness does not imply useful power for every finite-sample alternative.

| m | Detections / 20 independent experiments | 95% power interval |
|---:|---:|---|
| 100 | 1/20 | [0.009, 0.236] |
| 250 | 0/20 | [0.000, 0.161] |
| 500 | 0/20 | [0.000, 0.161] |
| 1000 | 0/20 | [0.000, 0.161] |
| 2000 | 4/20 | [0.081, 0.416] |

For these known localized laws, the decision rule selecting B when the center lies in {2,...,101} has optimal balanced accuracy 149/199 (about .7487). Thus weak MMD power here is not an information-theoretic impossibility of distinguishing the two artificial generators. The easy point-mass control and this localized control answer different sensitivity questions. A nonrejected learned law may still differ substantially in TV; no equivalence conclusion follows from these primary tests.
