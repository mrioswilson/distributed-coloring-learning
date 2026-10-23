# Exploratory candidate-channel intervention

This is separate from the frozen original-policy confirmation. It responds
to the requested mechanistic analysis and does not change a checkpoint,
primary hypothesis, graph, event, cutoff, sample count or test component.
It is explicitly an intervened policy diagnostic, not a fifth architecture
or a newly trained network. Original confirmatory observations are preserved.

## Controlled policy change

For each of the fifteen checkpoints, on all three new regular n512-d100 graphs
and the original historical graph (60 cells), replace only its candidate
channel by the exact local Luby `teacher_offer` on the policy's own current
state. Its learned participation, including the original float32 conversion
of r_p, and its original `LocalPaletteAdapter.commit_logits` are unchanged.
No exact COMMIT mask, safety repair, new authorization or termination rule is
introduced. Candidate replacement includes the teacher's original float64
uniform-rank convention. Checkpoint bytes remain identical. Unsafe decisions,
if any, are observed after they are fixed and are not corrected.

Couple the intervened policy to a separate exact Luby teacher using identical
per-node, per-round private uniforms in each pair. Distinct pairs have
independent namespaced PCG64 streams. Store both complete vectors, statuses,
rounds, seed and the indicator that their coarsened outcomes differ. A coarsened
outcome retains the full labeled vector on proper completion and one separate
atom for each other status. These paired observations are never fed into an
independent two-sample permutation test.

## Resource-only sample rule

Run after the original confirmation and planned trajectory diagnostics, sharing
the same single-GPU lock and the total 43,200-second ledger. A separate pilot
uses 16 pairs on the first new target graph for checkpoint41000 of each
architecture. Choose the first of m=1000 or500 pairs per cell for which 1.25
times the pilot-based total projected cost plus180 seconds fits the remaining
budget. Use only timing, never mismatch counts or event results, for that
choice. Save the sample freeze before drawing the new confirmatory pairs of
this exploratory intervention. If neither fits, retain the pilot and report
the intervention as skipped for resources. Do not enlarge samples after
reading results. Technical errors and hard-budget interruptions remain explicit.

Before this extra phase, conservatively reconcile the shared ledger against
the original process wall spans, including60-second startup/teardown margins
for each original pilot and the combined original confirmation/diagnostic
workflow. Add any uncharged housekeeping time. The intervention has its own
whole-process wall-span reconciliation and leaves150 seconds before the hard
cap for bookkeeping and context teardown. These extra charges do not change
any inferential code or original outcomes.

## Inference from a coupling

For any coupling of two laws on the same coarsened space,

    TV(P,Q) <= Pr(the coupled outputs differ).

For proper-completion probabilities p,q>0 and a full coarsened TV bound delta,

    TV(P(.|proper),Q(.|proper)) <= min(1,2 delta/min(p,q)).

For an event inside the proper subset, subtract the two normalized probabilities;
bound the unnormalized difference and the difference of normalizers by delta.
Taking the supremum proves the displayed conservative inequality.

Use exact binomial CP bounds simultaneously over 60 times5 probabilities:
mismatch, each generator's completion, and the intervened policy's two
preregistered terminal events A={U<=100}, T={color100 occurs}. Allocate total
failure probability .025. A conditional TV upper bound at most .05 supports
approximate equivalence **of the intervened policy and Luby**, not of the
original network and Luby. A large bound does not demonstrate a large distance:
the chosen common-random-number coupling can be loose.

Original event probability bounds already have a simultaneous .025 failure
budget over their elementary family. Combining these with the intervention
bounds gives joint coverage at least .95. The triangle inequality yields

    TV(original,intervened) >=
       original event-TV lower - intervened-versus-Luby TV upper.

The original-minus-intervened probability differences for A and T also give
direct TV lower bounds with that joint coverage. Reporting the maximum of
these simultaneously valid lower bounds is permitted. All 60 cells remain
visible, including failures to support restoration or a causal effect.

This is evidence about a specified computational intervention on the proposal
channel. It does not identify degree versus palette causality, establish
universal neural termination, or assert that an observed local bias alone
explains every original-law difference. No intervened result enters the 720
primary rejections or is described as a new original-policy confirmation.
