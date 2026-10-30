# Prospective CF confirmation, 30 October 2026

This protocol is fixed before scientific training and final evaluation. It is a
new experiment, not a replacement for any historical run. Configuration:
`configs/paper-revision-20261030.json`; namespace `revision-cf-types-20261030-v1`.
BE/E1 and speculative GM/E1 use the 19,396-parameter native-context architecture.
Ten fresh paired seeds per teacher, two arms, forty models. CPU, one worker.
The 1,500-update limit is fixed from a technical timing probe (100 synthetic
updates in 1.85 s), not from final accuracy. No milestone selection or retries.

## Split and acquisition

`paperrevision.keys.commit_key` describes sufficient actual queried-slot COMMIT
inputs, including phase/context, root features and a set of distinct neighbor
message types. MAX removes multiplicity; the key does not discard a COMMIT
input. Twenty percent of SHA256 buckets, with a fixed public salt, are held out
for every seed. No supervised COMMIT query in either training arm may belong to
those buckets. LINIAL and DETECT supervision are separate native decisions.
This is a split of sufficient input types, not a held-out intervention family,
and does not claim to exclude all more abstract rule-level similarities.

Each pair shares initialization and a bank of exactly 3,000 native component
targets. Native continuations failing the type filter are excluded. A fresh,
common auxiliary pool contains 3,600 rooted legal teacher-prefix states; both
native and intervened queries must pass the training-type filter before a state
is retained. Auxiliary targets are phase-balanced in GM (1,800 ASSIGN and
1,800 REPAIR); BE uses 3,600 KW targets. A single preassigned intervention per candidate state cycles the
root candidate through offsets 0,1,2,3, with 25% silence for offset 3. Other
messages stay native. This is a rooted typed-message intervention, not a joint
global teacher execution. The native arm acquires one native COMMIT target for
each retained auxiliary state; CF acquires its intervened COMMIT target. Thus
each model accesses 6,600 optimization targets, with identical base supervision
and auxiliary state distribution. Duplicate complete queries/types are counted.

Auxiliary native P/C conditioning messages are supplied equally to both arms;
they are not additional candidate/participation loss targets. Their acquisition
is **not free**: record 7,200 root values and all retained real-neighbor P/C
values separately. Report also simulator steps, queries considered/rejected,
generation wall/CPU time, distinct query/type counts and per-component budgets.
The comparison holds supervised-target and conditioning budgets equal; it does
not assert equal simulator effort or universal sample efficiency. Shared
physical acquisition is distinguished from per-model access.

## Optimization and freezing

Adam .001, batch 64, 1,500 updates, phases cycled, same index RNG in paired
arms. First 100 updates: native matched supervision. Later color phases:
`L_P + L_C + .5 L_native + .5 L_aux`, with a separate, phase-matched auxiliary
batch. LINIAL CE and DETECT BCE are unchanged. Auxiliary confirmation cannot
backpropagate into P/C decisions (messages are fixed); paired first-stage
weights are checked. Checkpoint every 250 updates for exact crash resumption;
only update 1,500 is analyzed. No development evaluation. Hash data, sources
and configuration before training, and all forty final weights before test.

## Final analysis

Only after the forty-model freeze generate 4,000 held-out continuation queries
per seed/teacher on fresh legal native prefixes, graph sizes 28/36, fresh PRNG
namespaces, and the same intervention templates. Record the natural class
balance within equal phase quotas (2,000 ASSIGN/REPAIR for GM, 4,000 KW for BE).
Record the natural class
balance; if a seed lacks either class, balanced accuracy is undefined and is
reported as such (no resampling based on labels). Primary endpoint is the mean
of the ten paired CF-minus-native balanced-accuracy differences per teacher.
Report all seed values and a percentile bootstrap interval over paired seeds
(10,000 resamples, fixed analysis seed); it is not an IID-query interval. Also
report occurrence and equal-per-type accuracy, confusion counts and
eligible/active/occupied strata. Zero train/test type intersection is asserted.

Evaluate each frozen model on 1,000 autonomous cases, sizes 32/48/64, common
case/tape banks within pairs; exact teacher reference first. No filters, rescue,
oracle, retries, or additional learning at student inference. Record success,
invariants, failures and native steps. These are secondary finite-domain
measurements; a local COMMIT gain need not improve autonomous progress.

The protocol's mathematical keys and code receive functional checks before
training. Independent human scientific review remains pending. Final results,
including unfavorable outcomes, are included without changing the protocol.
