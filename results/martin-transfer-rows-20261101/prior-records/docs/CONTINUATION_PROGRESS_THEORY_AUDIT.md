# Continuation, progress and repair: theory audit

Date: 31 October 2026. Status: explicit proofs and code correspondence checked
by the integrating agent; independent review remains pending. The integrator's
adversarial review is recorded separately. Finite executions below are checks of correspondence,
not proofs over all graphs, IDs, tapes or learned weights.

Release update: the newly supplied banks and weights permit full replication
of the additional studies' continuation decisions, and the historical raw
terminal colorings have now all been reconstructed. These checks are recorded
in `MARTIN_ARTIFACT_DELIVERY_AUDIT.md`; they neither change the propositions
below nor certify their assumptions for every learned policy. The original
proofs, diagnostic freezes and distinction between native and rollout domains
remain unchanged. Independent mathematical review is still pending.

## 1. Native-support underdetermination

Object: an admissible COMMIT input space X, native-message support N, and an
explicit hypothesis class H. If two rules in H agree throughout N but differ
on an admissible input outside N, native observations cannot select their
extension. Proof: their native input/label laws coincide. This is elementary;
it is not a new general learning lower bound, and a restrictive H or prior
can remove the ambiguity. A finite observed bank is not exhaustive N.

Implementation examples: BE KW, GM ASSIGN/REPAIR, and Greedy. Their native
participating roots are eligible and select a free option, hence native
confirmation equals root participation. Rules g0=p and g1=E*p*F agree there.
They differ on a legal active occupied-option query. The worked BE snapshot
has B7,q6,a_root6, neighbors3/0/1: option2 is free and option3 occupied.
GM/Greedy can query an already colored neighbor's color at a scheduled root.
The contract explicitly chooses g1. This example is not asserted for the
native Luby/Johansson collision support, where g0 need not agree.

`tests/test_integration_support_witness.py` additionally executes the entire
exact teacher prefix on a star with the coherent legal IDs (6,3,0,1).
Both Linial stages select point0 and preserve these constant-polynomial
values; earlier KW reductions leave them unchanged. Before native step13
the state is B7,q6 with exactly those colors. This establishes reachability
under an explicitly allowed ID assignment, not its frequency under a finite
PRNG seed battery or recovery of the historical witness's original IDs.

Correspondence: `src/dcl/multicf/oracles.py:confirmation`,
`src/dcl/multiteacher/programs.py:teacher_choices`,
`src/dcl/multiteacher_e1/supervision.py:labels_from_inputs`.

## 2. Gated transitions and KW necessity

Domain accounting: continuation-query banks use exact teacher-prefix states.
The frozen component protocol instead evaluates the same explicit KW predicate
on current reached states with valid auxiliary coloring/clock/bounds; such a
state need not follow the teacher's earlier choices. This diagnostic domain
extension is declared separately in the article and is not interpreted as
teacher-prefix holdout performance. The local transition proof below holds
for those legal states without needing an exact-teacher prefix.

For color phases only, with fixed state, messages and permissions, the update
is y'_v=D_v(S,c_v) if w_v*p_v*z_v=1, and y'_v=y_v otherwise. D is the
interpreter's decoding, not an availability oracle. Thus changing COMMIT at
silent/nonpermitted roots has no direct transition effect. At an active,
permitted root it changes the stored value iff decoding differs from the old
value. Eligibility is not enforced by the interpreter, so ineligible-active
errors can matter when participation errs. DETECT and LINIAL are excluded.

For KW, B becomes q=ceil(4B/5) regardless of writes. An old value a_v>=q
that does not write remains outside [0,q). This violates the checked bound
invariant immediately, regardless of neighbor writes. This necessity is for
the implemented shrinking-bound protocol, not for all coloring algorithms.

Proof: substitute the neural action decoder's P∧COMMIT action into the
interpreter's permission-gated assignment. Values at distinct vertices are
assigned from the same old state. Then apply the strict new interval.

Correspondence: `multiteacher_e1/model.py:native_prediction`,
`multiteacher/programs.py:ColoringProgram.apply` KW/color branches,
`samplecomplexity/evaluation.py:invariant_after`. The legal star snapshot in
`tests/test_coauthorintegration.py` checks B7→6 and a_root6 retained by a
missed write. The exact target selects option2; omission breaks the invariant.

## 3. GM: sufficiency, necessity and a counterexample

Sufficient: marks cover each speculative conflict edge and every queued root
writes a currently free color. For each initially conflicting edge, the last
repaired endpoint avoids the other; edges with no repaired endpoint were
already proper. Later free writes cannot create conflicts.

A skipped repair is not by itself sufficient for final failure. Sufficient
failure condition: a monochromatic edge survives the omission and neither
endpoint changes color in the remaining queue. Proof: both endpoint colors
then remain equal. A later repaired endpoint may instead clear it.

Legal counterexample: three one-vertex GM blocks on a path in increasing ID
order1,2,3 assign0 simultaneously; DETECT marks1 and2. Skip queued1, then
repair2 to1. Final (0,1,0) is proper. The test executes the actual teacher and
interpreter; it passes. This explicitly refutes the necessity of every
scheduled repair in every legal GM execution.

Correspondence: `ColoringProgram.snapshot/apply`, `teacher_choices` DETECT and
REPAIR, plus the second test in `tests/test_coauthorintegration.py`.

## 4. Progress with participation one

Define U_v=N(v)∩{pending vertices}; d_U(v)=|U_v|. Condition on the history
before fresh private draws. Assume independent participation/candidate pairs
across vertices and independence within each pair, candidate laws supported
on available lists A_u, and exact confirmation. Then

P(v writes|S)=p_v sum_c q_v(c) product_{u in U_v}(1-p_u*q_u(c)).

If q_v(c)<=beta/|A_v| and p_u*q_u(c)<=theta<1 on c in A_v, the log chord
log(1-x)>=log(1-theta)*x/theta and Jensen give the lower bound

p_v*(1-theta)^[beta*sum_u p_u/(theta*|A_v|)].

Proof: write Lambda_c=sum_u p_u*q_u(c). The product is at least exp(a*Lambda_c)
for a=log(1-theta)/theta<0. Jensen bounds its q_v average by
exp(a*sum_c q_v(c)*Lambda_c). The weighted load is at most
beta*sum_u p_u/|A_v|; multiplication by a reverses that inequality.
The empty neighborhood gives success p_v directly.

This includes the old p_max<1 result by theta=p_max and sum p_u<=p_max*d_U.
For always-active Johansson with uniform available proposals, each pending
neighbor has at least two available colors, so theta=1/2, beta=1, and
|A_v|>=d_U(v)+1 yields progress at least1/4. For arbitrary learned proposals,
theta<1 need not hold: identical deterministic proposals at adjacent
always-active vertices give zero progress. No checkpoint is claimed to
satisfy these assumptions uniformly. Submass and termination statements
retain their own uniformity and independence hypotheses.

## 5. Contract corrections

- Acquired oracles are explicitly K4. Variable-palette transfer is a separately
  declared extension, not arbitrary list-coloring support of those functions.
- BE interventions may change participation as well as the queried option.
- JP's additional equal-offer veto is a specified off-native extension,
  redundant for coherent native local maxima.
- Linial's native label selects the first unblocked point; correctness of a
  reduction with any unblocked point is a different statement.
- The worked KW negative uses stored active option3, and native transition13
  is distinguished from zero-based frame12. Native transition7 in the separate
  participation witness is the fifth KW reduction.
- Historical intervention generation is specified explicitly in the article,
  including silence conditional on variant3, rather than a misleading global
  silence probability.

The finite-type MAX proposition still requires all64 type subsets, legal
canonical inputs, nonnegative messages, exact gather and its numerical
assumptions. It does not cover the native residual BE/GM contexts or certify
progress. A failed sufficient certificate is not a reachable counterexample.

## Verification performed so far

`PYTHONPATH=src /workspace/paper-revision-20261030/.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_coauthorintegration.py`
returned **2 passed**. Both the KW necessity example and the GM counterexample
use actual state machines. Full manuscript and final review checks are
recorded in the integration handoff and review report.

Final checks: the selected regression suite passed330 tests; the additional
full-prefix support witness passed1. The 30,000 component executions and
15,000 historical-cell replays completed without technical errors, and the
six component baselines match all original terminal fields. A separate
2,000-case observer verifies84 identity-decoding contract errors, illustrating
the exact ``decode differs from old value'' qualification rather than
silently equating contract rejection with an unsafe state transition.
These executions support implementation correspondence; the proofs above
remain the mathematical justification. Independent review is still pending.
