# Sample complexity and learnability of distributed coloring

This release adds a separate, query-budgeted study to the preserved historical
experiments: seven native teachers, three architectures, five training seeds,
seven label budgets and 735 fresh primary trainings. Every primary training
uses 500 Adam updates. A package passes only when each of its five frozen
policies completes all 1,000 independent final executions correctly, with no
cutoff, invariant violation or technical error.

The primary result is **12 PASS and 9 NOT CERTIFIED packages**, with all 80,000
attempted student final runs retained. Luby passes at grid budgets 1,000 for
E1 NORMALIZED and LocalPaletteTransformer (999 acquired component labels per
seed), and 3,000 for DeepSets. Johansson, JP and Greedy pass at grid budget
1,000 for all three architectures. ColorTrials, BE and GM remain NOT CERTIFIED
under the primary protocol, including every observed negative result.

Separate exploratory experiments include small-model budget curves,
no-communication and interactive controls, complete development action and
transition audits, native-efficiency measurements and a robust global-law
test. The independent paired optimization experiment validates BE with E1 at
3,000 fixed labels and 1,500 updates in all five seeds; this is a fixed-budget
point, not a new minimum or a replacement for the primary 500-update result.

The runtime archive contains all 921 checkpoints, cached labeled examples,
optimizer/RNG states and raw development records: 3,004 files with an
individual SHA-256 manifest. Primary/final rows, frozen batteries, tables,
figures, diagnostics and the claim ledger are versioned in Git. The repository
includes verified fetch, evidence-check, numerical-training-replay and
independent-replication scripts. The paper, information audit, protocol and
scientific audit accompany this release.

The protocol was frozen locally before confirmatory training; it was not an
externally timestamped preregistration. PASS is a result about these frozen
policies and finite batteries. It is not universal correctness, universal
representability or a theoretical sample-complexity bound. The law supplement
has zero Holm rejections among 210 testable hypotheses; nonrejection does not
establish equality of coloring distributions.

Validation: 597 tests passed, all exact-teacher batteries passed their frozen
horizons, runtime/source/checkpoint hashes passed verification, the 30-page
paper compiled and was visually reviewed, and a fresh numerical training
replay reproduced the original weights exactly without new teacher queries.

Restore with `python scripts/fetch-sample-complexity.py`, then run
`python scripts/verify-sample-complexity.py --runtime`. See the README and
`docs/SAMPLE_COMPLEXITY_RESULTS.md` for reproduction commands and answers P1–P12.
