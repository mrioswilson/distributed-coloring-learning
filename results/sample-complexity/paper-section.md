# How Much Supervision Does It Take to Learn a Distributed Algorithm?

A teacher observation is one acquired decision label for a specified node,
state and action component. In ordinary color phases we count participation,
proposal and confirmation separately; Linial point selection and speculative
conflict marking each count as one. Replaying a stored label adds optimizer
work and no new supervision. Query counts, component types, trajectory clusters,
unique observations and data hashes are recorded independently of updates.
This distinction matters because sequential imitation changes the states a
policy visits [@ross2010efficient; @ross2011dagger]. Existing imitation
sample-complexity theory depends on its own state-space and horizon assumptions
[@rajaraman2020limits]; our finite curves imply no analogous theoretical bound.

Information compatibility is established before comparing budgets. Anonymous
regime A contains LUBY, JOHANSSON and repeated COLOR_TRIALS. Extended regime B
contains JP with explicit relative-ID comparisons, BE with auxiliary colors,
phase clock and supplied finite-field arithmetic, GM with priority comparisons
and its native block/repair schedule, and sequential GREEDY with writable-node
permissions. The three palette architectures have identical teacher-specific
observations and learned graph messages. Transformer attention acts within a
node's palette; it is not a replacement for graph communication. Interface
sufficiency arguments and an anonymous-JP indistinguishability example do not
prove universal representability by these fixed network widths.

For a policy to PASS, every frozen execution must end with a proper in-palette
coloring, all protocol invariants respected, termination by the teacher-specific
deadline, and no technical error, cutoff or incorrect terminal state. We use
no 95% functional threshold, repair filter or failed-tape replacement. A
teacher/architecture/budget combination passes only when all five independently
trained policies pass their respective 1,000-case final batteries. These are
five separate policies, not one policy tested 5,000 times.

All seven exact teachers passed every frozen final case before student
confirmation. Native deadlines distinguish n-step sequential greedy and JP's
at-most-n priority steps from BE's 15 reductions and the randomized teachers'
prospective tail deadlines. Native steps with different semantics are never
directly interpreted as identical distributed communication units.

The balanced cohort comprises 735 fresh trainings, five seeds per cell, seven query budgets from 1,000 to 1,000,000, and 500 Adam updates per training. Its shared teacher libraries acquired 34,999,967 component labels in total; smaller-budget policies only access their hashed prefixes. Training sizes are 8–24, development sizes 28/36, and confirmation sizes 32/48/64, in seven prescribed graph strata. The first budget where all five development policies are perfect nominates one frozen candidate; final results cannot choose another budget. Of 21 teacher/architecture packages, 16 nominated a candidate, 12 achieved five-seed PASS, and 80,000 final student executions were retained.

**Regime A. Smallest validated candidate budget, or explicit lack of certification.**

| Teacher | E1: B* | DeepSets: B* | Transformer: B* |
|---|---|---|---|
| LUBY | 1,000; 5/5 PASS | 3,000; 5/5 PASS | 1,000; 5/5 PASS |
| JOHANSSON | 1,000; 5/5 PASS | 1,000; 5/5 PASS | 1,000; 5/5 PASS |
| COLOR_TRIALS | NOT CERTIFIED; 3/5 (candidate 300,000) | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) |

**Regime B. Smallest validated candidate budget, or explicit lack of certification.**

| Teacher | E1: B* | DeepSets: B* | Transformer: B* |
|---|---|---|---|
| JP | 1,000; 5/5 PASS | 1,000; 5/5 PASS | 1,000; 5/5 PASS |
| BE_BASE_D3 | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) | NOT CERTIFIED; 0/5 confirmed (no candidate; max 1,000,000) |
| SPECULATIVE_GM | NOT CERTIFIED; 4/5 (candidate 30,000) | NOT CERTIFIED; 4/5 (candidate 30,000) | NOT CERTIFIED; 4/5 (candidate 30,000) |
| GREEDY | 1,000; 5/5 PASS | 1,000; 5/5 PASS | 1,000; 5/5 PASS |

A NOT CERTIFIED cell reports an unachieved protocol threshold B*>1,000,000,
not an impossibility theorem. Its earlier development budgets were evaluated;
policies failing development were not exhaustively tested in confirmation.
Per-seed queries, failures, compute and the distinction between an unattempted
candidate and a failed confirmation remain in the machine-readable tables.

![Anonymous-interface supervision curves. Each line is one training seed; these are development screens, not repeated final confirmations. The dashed line is the required 100% frontier.](ucgnn/figures/sample-success-regime-a.png)

![Extended-interface supervision curves with the same five-seed and 100% rules. Declines are retained without smoothing.](ucgnn/figures/sample-success-regime-b.png)

The grid does not assume monotonic improvement. Fixed optimization also means
larger libraries can leave more acquired observations unused; visit counts
are exported. Thus these are supervision curves conditional on the specified
optimizer budget, not pure information-theoretic sample requirements.

For zero failures in 1,000 independent runs of one frozen policy, the one-sided
95% binomial lower success bound is 0.9970087504549047. Under fixed independent
strata the same zero-failure bound conservatively controls their
allocation-weighted mean by the arithmetic–geometric mean inequality. It does
not cover every graph or simultaneously all trained policies. The five seeds
remain the units of training replication.

## Optimization, capacity and interactive controls

The following entries count seeds completing all development cases, at the
same B=3,000. They are exploratory development outcomes, never final PASS:

| Teacher | Passive E1 500 updates | Passive E1 1,500 updates | Small 500 | No messages 500 | Interactive 500 total |
|---|---|---|---|---|---|
| LUBY | 5/5 | 5/5 | 1/5 | 0/5 | 4/5 |
| JP | 5/5 | 5/5 | 0/5 | 0/5 | 5/5 |
| BE (D3) | 0/5 | 5/5 | 0/5 | 0/5 | 1/5 |

The fixed-label control changes only E1's optimizer updates. The small and
no-message models share the same label prefix and update budget. The
one-aggregation interactive control allocates half the labels to passive
warmup and half to admissible states visited by the warm policy, with 250+250
updates and an Adam reset at aggregation. Failed learner trajectories remain
recorded. It excludes GM's unproved arbitrary-history interface. These limited
controls cannot identify every failure as a representation or data limitation.

Functional PASS does not logically imply action or transition fidelity,
teacher-like execution time, or equality of complete coloring laws. Separate
teacher-state fidelity and native-step curves retain all five seeds. Historical
law tests are not imported as evidence for the new checkpoints. The full
protocol, audit, negative results, dataset/checkpoint hashes and reproducible
figure sources accompany this finite empirical statement.

The GM confirmation failures (normalized: 79, deepsets: 79, transformer: 79) leave only isolated vertices uncolored and occur in training seed(s) 83105. A post hoc singleton witness using the frozen candidate weights reproduces an incorrect confirmation rejecting assignment despite positive participation, while the exact teacher succeeds. The cached prefix contains 5 isolated-assignment bundles (three component labels per bundle), and the exact recorded minibatch order presents them 9 times; the affected seed's development suite contains 0 isolated vertices. This identifies a rare-state fit defect missed by development coverage. It does not show an absence of ID information or prove a universal capacity limitation. The diagnostic changes no model, budget, primary classification or final tape. Full traces and presentation-count checks are in structural-failure-witnesses.json.

The independent exploratory coloring-law supplement tests 210 fixed teacher/checkpoint/graph hypotheses using the audited native K4 robust instrument, split selection/evaluation rows, 9,999 permutations and Holm over the complete family. 210 comparisons are testable and 0 reject. Untestable comparisons retain p=1; conservative futility also cannot create rejection. These are conditional-on-success laws on two fixed small graphs, not equality guarantees or an additional functional PASS condition.

The small-model supplement repeats all seven budgets for three representative teachers and freezes independent final tapes under the same five-seed gate:

| Teacher | Small parameters | Small validated B* | Seeds confirmed | Final attempted |
|---|---|---|---|---|
| LUBY | 551 | NOT CERTIFIED | 0/5 | False |
| JP | 1560 | NOT CERTIFIED | 0/5 | False |
| BE_BASE_D3 | 1560 | NOT CERTIFIED | 0/5 | False |

Without a common development candidate no student final is attempted. These outcomes establish neither universal minimality nor the behavior of all possible small networks.

The additional fixed-label optimization protocol freezes both existing optimizer arms for all three control teachers and evaluates them on new paired final batteries. It checks identical label-prefix and initialization hashes and changes no primary outcome:

| Teacher | 500 updates, B=3,000 | 1,500 updates, B=3,000 |
|---|---|---|
| LUBY | PASS (5/5) | PASS (5/5) |
| JP | PASS (5/5) | PASS (5/5) |
| BE_BASE_D3 | NOT CERTIFIED (0/5) | PASS (5/5) |

A PASS here validates the fixed B=3,000/optimizer point only; it is not a new grid minimum or a retry of the primary confirmation.
