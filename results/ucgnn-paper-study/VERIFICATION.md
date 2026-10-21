# Verification and engineering observations

The full regression suite passed **257 tests** on 2026-10-21; exact command
and output are retained in `tests.txt`. It includes original tests and new
tests for tensor-identical initialization, isolated endpoint transformations,
exact NORMALIZED forward/gradient equivalence, real-data paired tapes and
epoch-boundary recovery, generic baselines, unshielded execution, independent
cohort planning, cached local states, realizable COMMIT stars and interval
enclosures compared with exact rational arithmetic.

A prior run during active GPU training had three lock-related failures
(`tests-during-training.txt`): the original tests require the same global
exclusive GPU lock already held by our worker. The suite passed after stopping
only our coordinator at an epoch boundary. This is not an algorithmic failure
or a change to historical code. Recovery events are recorded in
`data/ucgnn-paper-study/recovery-validation.json`; optimizer and RNG state
remain in the run's `last.pt`.

`resources-preflight.csv` is a resource-only test on separate deterministic
seed streams, with model decisions discarded. It uses inference mode and
does not consult a confirmatory outcome or select an architecture.
`resources-preflight-grad-enabled-debug.csv` preserves the earlier engineering
mistake of profiling with autograd enabled. Its high-degree OOMs concern
retained training graphs and **are not inference limits**. The corrected
inference-mode check fits every primary model at N=640, Delta=200, K=201.
Both profiling processes were charged to the accumulated GPU budget.

The original frozen protocol document is unchanged. Per-model diagnostic jobs
retain the same planned probes and the 1,800-second per-job limit. Splitting
them into sequential recoverable jobs avoids giving the entire mechanism
battery one timeout. This is task scheduling, not architecture selection.

Source files listed by the historical snapshot were rehashed and remained
unchanged. Some source CSVs have CRLF endings; the worktree's attributes preserve
their byte hashes instead of silently normalizing those snapshots on checkout.
