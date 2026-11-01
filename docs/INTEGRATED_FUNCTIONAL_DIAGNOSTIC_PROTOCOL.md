# Frozen-component BE diagnostic, 31 October 2026

This protocol precedes the diagnostic's replay outcomes. It is a retrospective
component intervention on consumed final cases, not a new final benchmark,
external preregistration, or training-method comparison. No model is trained,
tuned, selected for deployment, or substituted into historical results.

## Question and fixed selection

Which observed BE failures persist when only KW confirmation, participation,
or candidate selection is replaced? Select seeds105001,105008,105010 and both
native/CF arms from Pedro's40-model trial. These are all BE seeds with any
autonomous failure in that trial, selected using known historical outcomes.
The subset includes all2,595 observed CF failures and796 native failures;
these are paired observations, not independent training replicas or an
unbiased prevalence sample. Both arms and every one of their1,000 original
cases remain included, including successes.

## Interventions

For every selected model/case, run five modes:

1. `baseline`: identical learned decisions and the original interpreter.
2. `commit`: replace the KW confirmation bit by the declared oracle on the
   actual current learned participation/candidate messages; all other learned
   modules and Linial decisions remain unchanged.
3. `participation`: replace KW participation messages by permission and
   eligibility for every represented center; retain learned candidates and
   recompute learned second exchange/COMMIT on those messages.
4. `candidate`: replace KW candidate options by the first currently unoccupied
   option for every represented center; retain learned participation and
   recompute learned second exchange/COMMIT.
5. `kw_all`: exact KW participation, first free candidate and confirmation;
   learned Linial point decisions are retained.

All interventions are for diagnosis only and call exact rules; they are not
autonomous learned policies. The other modules retain their weights and are
recomputed on intervened trajectories. Their later numeric outputs need not
equal baseline outputs after states diverge. Cases, order, graph/tape seeds,
cohorts of32, inference batches of1024, float32 and one Torch thread per
process are preserved. No decisions are discarded. At most two independent
worker processes are used. The maximum is30,000 model-case executions.

## Hypotheses and recorded outcomes

- H1: COMMIT replacement rescues failures in CF105001/105008 more often than
  candidate replacement; their original held-out-query defect concerns a
  κ=0 rejection, not an occupied proposal.
- H2: participation replacement rescues more105010 failures than COMMIT
  replacement; the existing witness localizes an upstream omission.
- `kw_all` is a diagnostic bound conditional on unchanged learned Linial;
  failures, if any, are retained and investigated.

Record full final rows, success/failure/status/steps, per-case rescues and new
failures relative to baseline, and confusion strata of COMMIT on actual visited
inputs. Separate errors masked by root silence, unsafe accepts, missed valid
writes, eligibility-boundary errors, upstream silence and occupied proposals.
Counters from different trajectory modes have different visitation measures;
do not pool them as if they were common-state treatment effects.

Every baseline must reproduce all1,000 historical terminal colors, native steps,
statuses and invariant flags before its interventions are interpreted. Record
source/protocol/checkpoint hashes before execution and preserve them afterward.
No query-level significance, population rescue estimate or retrospective
coprimary endpoint is introduced. Any deviation is recorded before rerunning.
