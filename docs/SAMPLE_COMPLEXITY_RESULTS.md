# Sample complexity results and required scientific questions

**P1. How much Luby supervision?**

E1 NORMALIZED: B*=1,000, actual acquired labels 999 per seed, 5/5 PASS; DeepSets: B*=3,000, actual acquired labels 3000 per seed, 5/5 PASS; LocalPaletteTransformer: B*=1,000, actual acquired labels 999 per seed, 5/5 PASS. These are acquired component-label budgets under the frozen optimization/selection procedure, not trajectories or gradient updates.

**P2. How does the budget change across teachers?**

| Teacher | E1: B* | DeepSets: B* | Transformer: B* |
|---|---|---|---|
| LUBY | 1,000; 5/5 PASS | 3,000; 5/5 PASS | 1,000; 5/5 PASS |
| JOHANSSON | 1,000; 5/5 PASS | 1,000; 5/5 PASS | 1,000; 5/5 PASS |
| COLOR_TRIALS | NOT CERTIFIED; 3/5 (candidate 300,000) | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) |
| JP | 1,000; 5/5 PASS | 1,000; 5/5 PASS | 1,000; 5/5 PASS |
| BE_BASE_D3 | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) |
| SPECULATIVE_GM | NOT CERTIFIED; 4/5 (candidate 30,000) | NOT CERTIFIED; 4/5 (candidate 30,000) | NOT CERTIFIED; 4/5 (candidate 30,000) |
| GREEDY | 1,000; 5/5 PASS | 1,000; 5/5 PASS | 1,000; 5/5 PASS |

Regimes A/B must be interpreted separately; exact counts, per-seed component queries, failures and CPU costs are in `results-table.json`/`.csv`.

**P3. Which information adaptations are required?**

JP/GM receive edge order signs; BE receives auxiliary colors, phase/bound/target, local polynomial evaluation and raw equality relations; GM/greedy retain external scheduling. Luby/Johansson/ColorTrials use the anonymous regime. The full sufficiency/indistinguishability analysis is in `docs/TEACHER_MODEL_INFORMATION_COMPATIBILITY.md`.

**P4. Do adaptations resolve earlier JP/BE failures?**

JP: E1 NORMALIZED: B*=1,000, actual acquired labels 999 per seed, 5/5 PASS; DeepSets: B*=1,000, actual acquired labels 999 per seed, 5/5 PASS; LocalPaletteTransformer: B*=1,000, actual acquired labels 999 per seed, 5/5 PASS.

BE_BASE_D3: E1 NORMALIZED: NOT CERTIFIED, 0/5 confirmed; protocol threshold unachieved through 1,000,000; DeepSets: NOT CERTIFIED, 0/5 confirmed; protocol threshold unachieved through 1,000,000; LocalPaletteTransformer: NOT CERTIFIED, 0/5 confirmed; protocol threshold unachieved through 1,000,000.

These statements concern the new complete training/interface package. The previous integrated 70-model cohort already had numeric IDs and BE helpers. There is no matched raw-ID ablation at identical data/loss/updates, so a change cannot be attributed uniquely to the adapter. BE's arithmetic remains supplied. The additional fixed-label optimized BE point is reported separately in P8.

The GM confirmation failures (normalized: 79, deepsets: 79, transformer: 79) leave only isolated vertices uncolored and occur in training seed(s) 83105. A post hoc singleton witness using the frozen candidate weights reproduces an incorrect confirmation rejecting assignment despite positive participation, while the exact teacher succeeds. The cached prefix contains 5 isolated-assignment bundles (three component labels per bundle), and the exact recorded minibatch order presents them 9 times; the affected seed's development suite contains 0 isolated vertices. This identifies a rare-state fit defect missed by development coverage. It does not show an absence of ID information or prove a universal capacity limitation. The diagnostic changes no model, budget, primary classification or final tape. Full traces and presentation-count checks are in structural-failure-witnesses.json.

Across all recorded BE development runs, first-failure counts by stage are {'KW': 1563}. The separate common-state component audit at B=1,000,000 localizes agreement by phase and auxiliary-target sign; it does not identify a single causal failure mechanism.

**P5. Does training seed matter?**

All five curves are retained. There are 34 observed adjacent-budget success declines across individual seed curves; no monotonicity assumption is imposed. Per-cell `first_individual_development_pass`, `development_perfect_seeds` and final summaries expose variability, including when only a subset of seeds passes. Five seeds cannot characterize all possible training runs.

**P6. Which architecture needs least supervision?**

LUBY: E1 NORMALIZED, LocalPaletteTransformer at 1,000; JOHANSSON: E1 NORMALIZED, DeepSets, LocalPaletteTransformer at 1,000; COLOR_TRIALS: no certified ordering; JP: E1 NORMALIZED, DeepSets, LocalPaletteTransformer at 1,000; BE_BASE_D3: no certified ordering; SPECULATIVE_GM: no certified ordering; GREEDY: E1 NORMALIZED, DeepSets, LocalPaletteTransformer at 1,000. Ties and unachieved thresholds prevent a universal ranking. Earlier budgets that failed development were not all final-confirmed.

**P7. Are smaller models more sample efficient?**

At the same B=3,000 and 500 updates, these development-only controls compare matched label prefixes:

| Teacher | Passive E1 500 updates | Passive E1 1,500 updates | Small 500 | No messages 500 | Interactive 500 total |
|---|---|---|---|---|---|
| LUBY | 5/5 | 5/5 | 1/5 | 0/5 | 4/5 |
| JP | 5/5 | 5/5 | 0/5 | 0/5 | 5/5 |
| BE (D3) | 0/5 | 5/5 | 0/5 | 0/5 | 1/5 |

The separate small-model seven-budget grid has independently frozen confirmation batteries:

| Teacher | Parameters | Small validated B* | Seeds confirmed | Final attempted |
|---|---|---|---|---|
| LUBY | 551 | NOT CERTIFIED through 1,000,000 | 0/5 | False |
| JP | 1560 | NOT CERTIFIED through 1,000,000 | 0/5 | False |
| BE_BASE_D3 | 1560 | NOT CERTIFIED through 1,000,000 | 0/5 | False |

A missing development candidate means no student final was attempted; 0/5 confirmed does not mean five observed final failures.

Small-model conclusions apply to these matched training budgets and independent finite final batteries. Only one no-message budget is tested. Neither experiment establishes theoretical minimality.

The unsmoothed per-seed capacity comparison is exported as `results/sample-complexity/figures/capacity-comparison.png` and `.pdf`.

**P8. What changes with fixed labels and more optimization?**

Compare the 500/1,500-update columns above, with identical labels and initialization per seed. The complete control records retain per-seed development success and time. Improvements identify a finite optimization-budget sensitivity, not a universal explanation of every failure. No extra teacher query is charged for replay.

An additional exploratory protocol compares the same two frozen arms on independent, paired final batteries, after checking identical label-prefix and initialization hashes:

| Teacher | 500 updates, B=3,000 | 1,500 updates, B=3,000 |
|---|---|---|
| LUBY | PASS (5/5) | PASS (5/5) |
| JP | PASS (5/5) | PASS (5/5) |
| BE_BASE_D3 | NOT CERTIFIED (0/5) | PASS (5/5) |

These are fixed-budget validations, not new B* minima; they do not replace primary statuses.

For BE, the paired fixed-label control changes the number of fully successful seeds from 0/5 to 5/5 with more updates. This demonstrates optimization-budget sensitivity at those fixed labels, without estimating a minimum label budget for 1,500 updates.

**P9. Does interactive supervision help?**

The last column uses the same maximum B=3,000 and 500 total updates, with one 1,500-label passive / 1,500-label admissible on-policy aggregation. All seeds and failed acquisition trajectories are retained. Adam resets at aggregation; this control does not isolate distribution choice from every optimizer-history effect. It is development-only and supports no final-certification or general DAgger efficiency claim.

Observed development-perfect seed counts are LUBY: interactive 4/5 versus passive 5/5; JP: interactive 5/5 versus passive 5/5; BE_BASE_D3: interactive 1/5 versus passive 0/5. These fixed-point observations do not establish a universal modality ranking.

**P10. Does functional 100% imply action, time or global-law fidelity?**

No. The functional gate constrains validity, invariants and completion; it does not require matching teacher choices, native step counts or probabilities of each complete coloring. These are separate measured outcomes. The independent exploratory law supplement has 210/210 testable hypotheses and 0 Holm rejections; 0 rejections concern policies in functionally certified packages. Comparisons condition on successful rows; nonrejection never establishes equality.

**P11. What is robust across all five seeds?**

LUBY/normalized at 1,000, LUBY/deepsets at 3,000, LUBY/transformer at 1,000, JOHANSSON/normalized at 1,000, JOHANSSON/deepsets at 1,000, JOHANSSON/transformer at 1,000, JP/normalized at 1,000, JP/deepsets at 1,000, JP/transformer at 1,000, GREEDY/normalized at 1,000, GREEDY/deepsets at 1,000, GREEDY/transformer at 1,000. This is robustness across these five trainings and batteries, not a guarantee for unseen initializations or all graphs. Every policy retains its own 1,000-run uncertainty statement.

The separate full-fidelity audit covers every root and every transition of all 28 development teacher trajectories per seed, including late KW stages. Its actual-state transition equality allows redundant actions that leave state unchanged. Original 256-decision prefix diagnostics remain in training logs and are not treated as a complete phase/family audit.

**P12. What belongs in the main paper and appendix?**

Main text: information regimes, query unit, strict five-seed gate, teacher-specific horizons, primary B* table, unsmoothed success curves, and concise fixed-label/interactive controls. Appendix and repository: all per-seed/per-budget counts, phase fidelity, native-step curves, checkpoint/data manifests, negative trajectories, source/RNG hashes, law-test details and resource/replay audits. No censored or missing cell is filled by extrapolation.
